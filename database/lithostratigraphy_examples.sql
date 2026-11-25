-- All source lithostratigraphy, including wellbores absent from the geochemical dataset.
SELECT area,wellbore_name,unit_level,unit_name,top_depth_m,bottom_depth_m,thickness_m
FROM geochem.v_wellbore_lithostratigraphy
ORDER BY wellbore_name,top_depth_m,bottom_depth_m,unit_level;

-- Area comes from the existing well table; this filter excludes unmatched wellbores.
SELECT wellbore_name,unit_name,top_depth_m,bottom_depth_m,thickness_m
FROM geochem.v_wellbore_lithostratigraphy
WHERE area='Barents Sea' AND unit_level='FORMATION'
ORDER BY wellbore_name,top_depth_m;

-- Exact five-column Excel projection; order is the original workbook row order.
SELECT wellbore_name AS "Wellbore name",top_depth_m AS "Top depth [m]",
       bottom_depth_m AS "Bottom depth [m]",unit_name AS "Lithostrat. unit",unit_level AS "Level"
FROM geochem.v_wellbore_lithostratigraphy
WHERE import_batch_id=8
ORDER BY source_sheet,source_row;

-- Retained source ambiguity: same level, same wellbore, overlapping depths.
SELECT * FROM geochem.v_lithostrat_interval_overlap;

-- Wellbores requiring a separate mapping or more geographic metadata.
SELECT wellbore_name FROM geochem.lithostrat_wellbore WHERE well_id IS NULL ORDER BY wellbore_name;
