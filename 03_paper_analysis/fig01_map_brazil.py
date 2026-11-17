"""Paper Figure 1: reference map of Brazil.

Country border, IBGE macro-regions (filled), states and municipalities on a cartopy map with
coastlines and a latitude/longitude grid.

Inputs  (cfg.paths): shp_brazil, shp_regions, shp_states, municipal (cleaned municipal mesh
        ``geometry/municipalities_2024.gpkg``; falls back to the raw IBGE mesh ``shp_municipalities``)
Outputs (cfg.paths.figures): fig01_map_brazil.{pdf,png}

Usage
    python 03_paper_analysis/fig01_map_brazil.py [--test] [--language en|pt] [--highlight CD_MUN ...]

--test : no municipality boundaries (the slowest layer); output goes below the test root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from hazbra.config import load_config, add_common_args
from hazbra import plotting as hp

REGION_COLORS = {'N': '#B3DE69', 'NE': '#FDB462', 'CO': '#FFED6F', 'SE': '#80B1D3', 'S': '#BC80BD'}
REGION_NAMES = {
    'en': dict(hp.REGION_NAMES_EN),
    'pt': {'N': 'Norte', 'NE': 'Nordeste', 'CO': 'Centro-Oeste', 'SE': 'Sudeste', 'S': 'Sul'},
}
LEGEND_TEXT = {'en': ('States', 'Municipalities'), 'pt': ('Estados', 'Municípios')}
EXTENT = (-75, -33, -35, 6)
PC = ccrs.PlateCarree()


def load_layers(cfg, municipalities=True):
    """Read the vector layers (all reprojected to EPSG:4326)."""
    layers = dict(brazil=hp.brazil_gdf(cfg, 'dissolved'), regions=hp.get_regions(cfg),
                  states=hp.get_states(cfg), municipalities=None)
    if municipalities:
        # cleaned municipalities (islands and lagoons removed); falls back to the raw IBGE mesh
        clean = hp.input_path(cfg, 'municipal') / 'geometry' / 'municipalities_2024.gpkg'
        if clean.exists():
            layers['municipalities'] = gpd.read_file(clean, layer='municipalities').to_crs(4326)
        else:
            layers['municipalities'] = gpd.read_file(cfg.paths.shp_municipalities).to_crs(4326)
    return layers


def plot_brazil(layers, language='en', show_municipalities=True, label_states=True,
                label_regions=True, highlight=None, coastlines=True, context=True, grid_step=10,
                extent=EXTENT, figsize=(6.5, 6.5)):
    """
    language : 'en' (North, Northeast, Central-West, Southeast, South) or 'pt' (original names)
    highlight: list of CD_MUN codes to outline and label (e.g. ['5300108'] for Brasília)
    coastlines / context : Natural Earth coastlines / land and neighbouring borders
    grid_step: spacing of the latitude/longitude grid (degrees)
    """
    reg, st, mun, br = layers['regions'].copy(), layers['states'], layers['municipalities'], layers['brazil']
    names = REGION_NAMES.get(language, REGION_NAMES['en'])
    if 'SIGLA_RG' in reg.columns:
        fallback = reg['NM_REGIA'] if 'NM_REGIA' in reg.columns else reg['SIGLA_RG']
        reg['label'] = reg['SIGLA_RG'].map(names).fillna(fallback)
        colors = [REGION_COLORS.get(c, '0.85') for c in reg['SIGLA_RG']]
    else:
        reg['label'] = reg['NM_REGIA'] if 'NM_REGIA' in reg.columns else reg.index.astype(str)
        colors = ['0.85'] * len(reg)

    fig = plt.figure(figsize=figsize, layout='constrained')
    ax = fig.add_subplot(1, 1, 1, projection=PC)
    ax.set_extent(extent, crs=PC)

    if context:
        ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='none', zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='0.55', linewidth=0.4, zorder=1)

    # regions (fill)
    reg.plot(ax=ax, color=colors, edgecolor='none', alpha=0.55, zorder=2)

    # municipalities, states, regions, country (lines, thin to thick)
    if show_municipalities:
        mun.boundary.plot(ax=ax, color='0.45', linewidth=0.05, zorder=3)
    st.boundary.plot(ax=ax, color='0.25', linewidth=0.45, zorder=4)
    reg.boundary.plot(ax=ax, color='0.1', linewidth=0.9, zorder=5)
    br.boundary.plot(ax=ax, color='k', linewidth=1.1, zorder=6)
    if coastlines:
        ax.coastlines('50m', color='0.2', linewidth=0.5, zorder=6)

    if label_states:
        lab = 'SIGLA_UF' if 'SIGLA_UF' in st.columns else ('NM_UF' if 'NM_UF' in st.columns else None)
        if lab:
            for _, r in st.iterrows():
                p = r.geometry.representative_point()
                ax.text(p.x, p.y, r[lab], ha='center', va='center', fontsize=6, color='0.15',
                        transform=PC, zorder=7)

    if label_regions:
        for _, r in reg.iterrows():
            p = r.geometry.representative_point()
            ax.text(p.x, p.y - 1.2, str(r['label']).upper(), ha='center', va='center', fontsize=7.5,
                    fontweight='bold', color='0.1', alpha=0.75, transform=PC, zorder=7)

    if highlight:
        if mun is None:
            raise ValueError('highlight needs the municipality layer (do not use --test)')
        hl = mun[mun['CD_MUN'].astype(str).isin([str(c) for c in highlight])]
        hl.plot(ax=ax, facecolor='none', edgecolor='#D7301F', linewidth=1.0, zorder=8)
        for _, r in hl.iterrows():
            p = r.geometry.representative_point()
            ax.plot(p.x, p.y, 'o', ms=3, color='#D7301F', transform=PC, zorder=9)
            ax.text(p.x + 0.4, p.y + 0.4, r['NM_MUN'], fontsize=6.5, color='#D7301F', fontweight='bold',
                    transform=PC, zorder=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # latitude / longitude grid
    gl = ax.gridlines(crs=PC, draw_labels=True, linewidth=0, color='0.5', alpha=0.6, linestyle='-')   # lines hidden, labels only
    gl.top_labels = gl.right_labels = False
    lon0 = int(grid_step * round(extent[0] / grid_step))
    lat0 = int(grid_step * round(extent[2] / grid_step))
    gl.xlocator = FixedLocator(list(range(lon0, int(extent[1]) + 1, grid_step)))
    gl.ylocator = FixedLocator(list(range(lat0, int(extent[3]) + 1, grid_step)))
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {'size': 7, 'color': '0.25'}

    s_lab, m_lab = LEGEND_TEXT.get(language, LEGEND_TEXT['en'])
    handles = [mpatches.Patch(color=c, alpha=0.55, label=n) for c, n in zip(colors, reg['label'])]
    handles.append(Line2D([], [], color='0.25', lw=0.45, label=s_lab))
    if show_municipalities:
        handles.append(Line2D([], [], color='0.45', lw=0.3, label=m_lab))
    ax.legend(handles=handles, loc='lower right', frameon=True, framealpha=0.9, edgecolor='none',
              fontsize=7)
    return fig, ax


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(ap)
    hp.add_plot_args(ap)
    ap.add_argument('--language', choices=['en', 'pt'], default='en', help='region names (default: en)')
    ap.add_argument('--highlight', nargs='*', default=None, metavar='CD_MUN',
                    help='municipality codes to outline and label (e.g. 5300108 = Brasília)')
    args = ap.parse_args(argv)

    hp.init_backend(args.show)
    cfg = load_config(test=args.test)
    show_mun = not args.test
    layers = load_layers(cfg, municipalities=show_mun)
    fig, _ = plot_brazil(layers, language=args.language, show_municipalities=show_mun,
                         highlight=args.highlight)
    hp.save_figure(fig, 'fig01_map_brazil', cfg.paths.figures, args.formats, dpi=300)
    hp.finish_figure(fig, args.show)


if __name__ == '__main__':
    main()
