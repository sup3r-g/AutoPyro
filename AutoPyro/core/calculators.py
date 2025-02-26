"""Calculators - static relations with parameters, don't depend on anything."""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from AutoPyro.core.base import BaseCalculator, Label
from AutoPyro.core.geometries import (
    LabelCurve,
    LabelPoint,
    average_curves,
    minimal_distances,
)
from AutoPyro.core.charts import Chart


class TransformationRatio:
    # Transformation Ratio
    COLUMN_NAME = "TR"

    class Espitalie_1987(BaseCalculator):
        def __call__(
            self,
            HI: float | npt.NDArray[np.float_],
            HI_o: float | npt.NDArray[np.float_],
        ) -> float | npt.NDArray[np.float_]:
            return (HI_o - HI) * 1200 / HI_o * (1200 - HI)

    class Peters_2005(BaseCalculator):
        def __call__(
            self,
            HI: float | npt.NDArray[np.float_],
            HI_o: float | npt.NDArray[np.float_],
            TOC: float | npt.NDArray[np.float_],
            TOC_o: float | npt.NDArray[np.float_],
            p: float = 83.33,
        ) -> float | npt.NDArray[np.float_]:
            return 1 - (HI * TOC * (p - TOC_o)) / (HI_o * TOC_o * (p - TOC))


class HIo:
    # Original Hydrogen Index
    COLUMN_NAME = "HIo"
    SELECTOR = "MATTER TYPE"

    class Cornford_2001(BaseCalculator):
        def __call__(
            self,
            HI: float | npt.NDArray[np.float_],
            T_max: float | npt.NDArray[np.float_],
        ) -> float | npt.NDArray[np.float_]:
            return HI + HI * (T_max - 435) / 30

    class Chart(BaseCalculator):
        def __call__(
            self,
            HI: float | npt.NDArray[np.float_],
            T_max: float | npt.NDArray[np.float_],
            author: str,
        ) -> list[Any]:
            plot = Chart.from_author(author)
            labelled_points = [
                LabelPoint(x, y, Label("", ""))
                for x, y in zip(T_max, HI)
                # if not np.isnan(x) and not np.isnan(y)
            ]
            hi_curves = sorted(
                [curve for curve in plot.curves if curve.label.name == HIo.SELECTOR],
                # Value with maximum y coordinate
                key=lambda curve: curve.coords[0][1],
                reverse=True,
            )

            # Add 0 line to analyze 3 type
            if len(hi_curves) <= 3:
                target_coords = hi_curves[-1].coords
                hi_curves.append(
                    LabelCurve(
                        [
                            (x, 0)
                            for x in np.linspace(
                                target_coords[0][0],
                                target_coords[-1][0],
                                len(target_coords),
                            )
                        ],
                        Label(hi_curves[-1].label.name, "IV"),
                    )
                )

            # Now transform curves into geometries
            ratios = np.linspace(0, 1, 25 + 1)
            curves_map = []
            for one, two in zip(hi_curves[1:], hi_curves[:-1]):
                curves_map.extend(list(average_curves(one, two, ratios).values()))

            return [
                curves_map[i].points()[0]
                for i in minimal_distances(labelled_points, curves_map)
            ]


class TOCo:
    # Original Total Organic Carbon
    COLUMN_NAME = "TOCo"
    # SELECTOR = "MATTER TYPE"

    class Peters_2005(BaseCalculator):
        def __call__(
            self,
            HI: float | npt.NDArray[np.float_],
            HI_o: float | npt.NDArray[np.float_],
            TOC: float | npt.NDArray[np.float_],
            TR: float | npt.NDArray[np.float_],
            p: float = 83.33,
        ) -> float | npt.NDArray[np.float_]:
            return p * HI * TOC / (HI_o * (1 - TR / 100) * (p - TOC) + HI * TOC)

    class Neruchev_1998(BaseCalculator):
        NERUCHEV_TABLE = pd.read_json(
            """
            {
                "columns":["Ro","Tmax","Градация катагенеза","Тип 1","Тип 2","Тип 3"],
                "data":[
                    [0.25,401,"ПК1",1.03,1.03,1.08],
                    [0.3,405,"ПК2",1.03,1.03,1.08],
                    [0.4,412,"ПК3",1.03,1.03,1.08],
                    [0.5,430,"MK1",1.22,1.22,1.09],
                    [0.65,438,"МК2",1.43,1.43,1.1],
                    [0.85,448,"МК3",2.32,2.32,1.19],
                    [1.15,459,"МК4",2.66,2.66,1.21],
                    [1.5,475,"МК5",null,null,1.22],
                    [2.0,530,"AK1",3.01,3.01,1.23],
                    [2.5,566,"АК2",3.16,3.16,1.26],
                    [3.5,650,"АК3",3.23,3.23,1.31],
                    [7.0,900,"АК4",3.26,3.26,1.33],
                    [11.0,1196,"Графит",3.27,3.27,1.43]
                ]
            }
            """,
            orient="split",
        )

        def __call__(
            self,
            TOC: float,
            organic_matter: str = "Тип 1",
            Ro: Optional[float] = None,
            Tmax: Optional[float] = None,
            maturity_level: Optional[str] = None,
        ) -> float:
            table = self.NERUCHEV_TABLE
            idx = table.index
            if maturity_level is not None:
                na_filter = maturity_level.notna()
                filtered = maturity_level.map(
                    {
                        v: int(k)
                        for k, v in table["Градация катагенеза"].to_dict().items()
                    }
                )
            elif Ro is not None:
                bins = table["Ro"].to_list() + [np.inf]
                filtered = pd.cut(
                    Ro, bins, labels=idx, include_lowest=True, right=False
                )
            elif Tmax is not None:
                bins = table["Tmax"].to_list() + [np.inf]
                filtered = pd.cut(
                    Tmax, bins, labels=idx, include_lowest=True, right=False
                )

            result = pd.Series(index=TOC.index, dtype=np.float64)
            if isinstance(organic_matter, pd.Series):
                om_format = "Тип " + organic_matter.apply(lambda x: f"{x:.0f}")
                om_keep = om_format.isin(["Тип 1", "Тип 2", "Тип 3"])
                locator = zip(filtered[filtered.notna()], om_format[om_keep])
                multiply_array = table.stack(dropna=False).loc[locator].to_numpy()

                print(multiply_array.shape, TOC[filtered.notna() & om_keep].shape)

                result.loc[filtered.notna() & om_keep] = (
                    TOC[filtered.notna() & om_keep] * multiply_array
                )
                result.loc[filtered.isna() | ~om_keep] = np.nan
            else:
                result.loc[filtered.notna()] = (
                    TOC[filtered.notna()]
                    * table.loc[filtered[filtered.notna()], organic_matter].to_numpy()
                )
                result.loc[filtered.isna()] = np.nan

            return result


CALCULATORS_MAP = {"HIo": HIo, "TOCo": TOCo, "TR": TransformationRatio}

if __name__ == "__main__":
    print(TransformationRatio.COLUMN_NAME)
    print(HIo.Cornford_2001()(600, 430))
