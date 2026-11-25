"""Persistent raster layers and north-up GeoTIFF export for interpolation grids."""
import ctypes
import json
from ctypes.util import find_library
from functools import lru_cache
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import xml.etree.ElementTree as ET

import numpy as np

from .mapping import crs, transformer


DIFFERENCE_METHOD = 'TIN-IDW'
SPREAD_METHOD = 'IDW-SPREAD'
TIN_SPREAD_METHOD = 'TIN-SPREAD'
KRIGING_SPREAD_METHOD = 'KRIGING-SPREAD'
SPREAD_METHODS = (SPREAD_METHOD, TIN_SPREAD_METHOD, KRIGING_SPREAD_METHOD)
COVERAGE_CODES = {0: 'Neither method', 1: 'IDW only', 2: 'TIN only', 3: 'Both methods'}
COVERAGE_COLORS = ['#d2d2d2', '#e5a400', '#00876c', '#4f73ae']


def surface_bands(result, method):
    grid = result.get('grid', result) if method == 'IDW' else result
    inside = np.asarray(grid['inside_domain'], dtype=bool)
    if method == DIFFERENCE_METHOD:
        coverage = np.zeros(inside.shape, dtype=float)
        for code, key in enumerate(('idw_only', 'tin_only', 'common'), 1):
            coverage[grid[key]] = code
        return np.asarray(grid['x']), np.asarray(grid['y']), {
            'difference': np.where(inside, grid['difference'], np.nan),
            'coverage': np.where(inside, coverage, np.nan),
        }
    bands = {}
    keys = ('spread', 'count') if method in SPREAD_METHODS else ('prediction', 'neighbors', 'nearest_distance_m')
    for key in keys:
        values = np.asarray(grid[key], dtype=float)
        bands[key] = np.where(inside & np.isfinite(values), values, np.nan)
    return np.asarray(grid['x']), np.asarray(grid['y']), bands


def compare_idw_tin_surfaces(idw, tin):
    if any(not np.array_equal(idw[key], tin[key]) for key in ('x', 'y', 'inside_domain')):
        raise ValueError('Для сравнения поверхностей нужна общая сетка и маска области.')
    inside = np.asarray(idw['inside_domain'], dtype=bool)
    a = np.where(inside, np.asarray(idw['prediction'], dtype=float), np.nan)
    b = np.where(inside, np.asarray(tin['prediction'], dtype=float), np.nan)
    valid_a, valid_b = np.isfinite(a), np.isfinite(b)
    common = valid_a & valid_b
    difference = np.full(a.shape, np.nan)
    np.subtract(b, a, out=difference, where=common)
    return dict(idw=a, tin=b, difference=difference, common=common,
                idw_only=valid_a & ~valid_b, tin_only=valid_b & ~valid_a,
                neither=inside & ~valid_a & ~valid_b)


def finite_surface_range(*arrays):
    limits = []
    for array in arrays:
        finite = array[np.isfinite(array)]
        if finite.size:
            limits.append((float(finite.min()), float(finite.max())))
    if not limits:
        return None
    return min(low for low, _ in limits), max(high for _, high in limits)


def _edges(centers):
    step = float(centers[1] - centers[0])
    return np.r_[centers - step / 2, centers[-1] + step / 2]


def raster_coordinates(layer, target):
    raster = layer['raster']
    x, y = np.meshgrid(_edges(np.asarray(raster['x'])), _edges(np.asarray(raster['y'])))
    if crs(layer['crs']) == crs(target):
        return x, y
    return transformer(layer['crs'], target).transform(x, y, errcheck=True)


def surface_layers(result, method, source_crs, name, labels, unit):
    x, y, bands = surface_bands(result, method)
    layers = []
    for key, values in bands.items():
        if not np.isfinite(values).any():
            continue
        band_unit = unit if key in ('prediction', 'difference', 'spread') else 'm' if key == 'nearest_distance_m' else ''
        display = {}
        if key == 'difference':
            limit = max(map(abs, finite_surface_range(values))) or 1
            display = dict(limits=[-limit, limit])
        elif key == 'coverage':
            display = dict(categories=labels.get('coverage_categories', list(COVERAGE_CODES.values())),
                           colors=COVERAGE_COLORS)
        elif key == 'count':
            display = dict(limits=[0, result['blocks']])
        layers.append(dict(id=str(uuid4()), name=f'{name} · {labels[key]}', crs=crs(source_crs).to_wkt(),
            features=[], attributes=[], fields=[], visible=key in ('prediction', 'difference', 'spread'), color='#333333', fill='#cccccc',
            alpha=.85, line_width=1, cmap='RdBu_r' if key == 'difference' else 'viridis',
            raster=dict(x=x.tolist(), y=y.tolist(), values=np.where(np.isfinite(values), values, None).tolist(),
                        band=key, unit=band_unit, method=method, options=result.get('options', {}),
                        model=result.get('model'), validation=result.get('validation'), **display)))
    if (method == DIFFERENCE_METHOD or method in SPREAD_METHODS) and len(layers) == 1:
        layers[0]['visible'] = True
    return layers


@lru_cache(maxsize=1)
def _gdal_library():
    # Importing Pyogrio loads and configures its GDAL and PROJ libraries.
    from pyogrio import _ogr

    extension = Path(_ogr.__file__)
    if os.name == 'nt':
        candidates = sorted((extension.parent.parent / 'pyogrio.libs').glob('gdal*.dll'))
    else:
        # POSIX symbol lookup on the extension includes its linked dependencies.
        candidates = [extension]
    for candidate in [*candidates, None]:
        if candidate is None:
            candidate = find_library('gdal')
            if not candidate:
                continue
        try:
            library = ctypes.CDLL(str(candidate))
            library.GDALCreateCopy
        except (OSError, AttributeError):
            continue
        return library
    raise ValueError('Не удалось загрузить библиотеку GDAL для экспорта GeoTIFF.')


def _copy_geotiff(vrt, output):
    library = _gdal_library()
    pointer, text, integer = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
    api = {
        'GDALAllRegister': (None, []),
        'GDALOpen': (pointer, [text, integer]),
        'GDALGetDriverByName': (pointer, [text]),
        'GDALCreateCopy': (pointer, [pointer, text, pointer, integer, ctypes.POINTER(text), pointer, pointer]),
        'GDALClose': (None, [pointer]),
        'CPLErrorReset': (None, []),
        'CPLGetLastErrorMsg': (text, []),
        'CPLGetLastErrorType': (integer, []),
    }
    for name, (result, arguments) in api.items():
        function = getattr(library, name)
        function.restype = result
        function.argtypes = arguments

    def failure():
        detail = library.CPLGetLastErrorMsg().decode('utf-8', errors='replace')
        return ValueError(detail or 'Не удалось создать GeoTIFF.')

    library.GDALAllRegister()
    library.CPLErrorReset()
    driver = library.GDALGetDriverByName(b'GTiff')
    if not driver:
        raise failure()
    source = library.GDALOpen(str(vrt).encode('utf-8'), 0)
    if not source:
        raise failure()
    try:
        options = (text * 2)(b'COMPRESS=DEFLATE', None)
        target = library.GDALCreateCopy(driver, str(output).encode('utf-8'), source, 1, options, None, None)
        if not target:
            raise failure()
        library.GDALClose(target)
        if library.CPLGetLastErrorType() >= 3:
            raise failure()
    finally:
        library.GDALClose(source)


def export_geotiff(path, result, method, source_crs, unit):
    x, y, bands = surface_bands(result, method)
    width, height = len(x), len(y)
    xe, ye = _edges(x), _edges(y)
    transform = [xe[0], x[1] - x[0], 0, ye[-1], 0, y[0] - y[1]]
    root = ET.Element('VRTDataset', rasterXSize=str(width), rasterYSize=str(height))
    ET.SubElement(root, 'SRS').text = crs(source_crs).to_wkt()
    ET.SubElement(root, 'GeoTransform').text = ','.join(map(str, transform))
    metadata = ET.SubElement(root, 'Metadata')
    ET.SubElement(metadata, 'MDI', key='INTERPOLATION').text = method
    for key, field in [('OPTIONS_JSON', 'options'), ('MODEL_JSON', 'model'), ('VALIDATION_JSON', 'validation')]:
        if result.get(field) is not None:
            ET.SubElement(metadata, 'MDI', key=key).text = json.dumps(result[field], ensure_ascii=False, allow_nan=False)
    ET.SubElement(metadata, 'MDI', key='VALUE_UNIT').text = unit
    if result.get('created_at'):
        ET.SubElement(metadata, 'MDI', key='CREATED_AT').text = result['created_at']
    for key in ('validation_note', 'uncertainty_method'):
        if result.get(key):
            ET.SubElement(metadata, 'MDI', key=key.upper()).text = result[key]
    if 'locations' in result:
        ET.SubElement(metadata, 'MDI', key='LOCATION_COUNT').text = str(len(result['locations']))
    if method in SPREAD_METHODS:
        for key, value in [('BLOCKS', result['blocks']), ('BLOCK_SIZE_KM', result['options']['block_size_km']),
                           ('BLOCK_SHIFT_KM', result['options']['block_shift_km']),
                           ('BUFFER_KM', result['options']['validation_buffer_km'])]:
            ET.SubElement(metadata, 'MDI', key=key).text = str(value)
        if result.get('model') is not None:
            ET.SubElement(metadata, 'MDI', key='KRIGING_MODEL').text = json.dumps(result['model'], allow_nan=False)
    path = Path(path)
    with TemporaryDirectory(prefix='autopyro-geotiff-', dir=path.parent) as temp:
        work = Path(temp)
        for number, (key, values) in enumerate(bands.items(), 1):
            raw = f'{key}.raw'
            np.asarray(values[::-1], dtype='<f8').tofile(work / raw)
            band = ET.SubElement(root, 'VRTRasterBand', dataType='Float64', band=str(number), subClass='VRTRawRasterBand')
            tags = dict(Description=key, NoDataValue='nan', ImageOffset='0', PixelOffset='8',
                        LineOffset=str(width * 8), ByteOrder='LSB')
            for tag, value in tags.items():
                ET.SubElement(band, tag).text = value
            ET.SubElement(band, 'SourceFilename', relativeToVRT='1').text = raw
            if key == 'coverage':
                categories = ET.SubElement(band, 'Metadata')
                for code, label in COVERAGE_CODES.items():
                    ET.SubElement(categories, 'MDI', key=f'CODE_{code}').text = label
            band_unit = unit if key in ('prediction', 'difference', 'spread') else 'm' if key == 'nearest_distance_m' else ''
            if band_unit:
                ET.SubElement(band, 'UnitType').text = band_unit
        vrt, output = work / 'surface.vrt', work / 'surface.tif'
        ET.ElementTree(root).write(vrt, encoding='utf-8', xml_declaration=True)
        _copy_geotiff(vrt, output)
        output.replace(path)
