"""
Recoleccion de POIs para TODOS los proyectos, en paralelo controlado --
version .py de recoleccion_masiva_con_filtros.ipynb, para correr desde
VS Code/terminal en vez de notebook.

Misma logica exacta: distancia caminando real (red vial + Dijkstra + KDTree),
filtros de calidad (nombres vacios, colegios duplicados por nombre,
universidades-institutos, limite de 1000m caminando, Tambo >50m).

Archivos esperados en la MISMA carpeta:
  - proyectos_geocodificados.xlsx

Genera (con checkpoint, retoma automaticamente si se corta):
  - pois_detalle_por_proyecto.csv
  - pois_conteos_por_proyecto.csv
  - pois_ids_procesados.csv
"""
import os, sys, time, math, unicodedata, re, threading, heapq
import numpy as np, pandas as pd
import requests
from scipy.spatial import cKDTree
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════ CONFIG ═══════════════════════════
RUTA_PROYECTOS = 'proyectos_geocodificados.xlsx'
RUTA_DETALLE   = 'pois_detalle_por_proyecto.csv'
RUTA_CONTEOS   = 'pois_conteos_por_proyecto.csv'
RUTA_IDS_OK    = 'pois_ids_procesados.csv'

RADIO_M                  = 1000
BBOX_RADIO_RED_VIAL      = RADIO_M + 500
UMBRAL_SNAP_M            = 200
LIMITE_DIST_CAMINANDO_M  = 1000
UMBRAL_SIMILITUD_NOMBRE  = 0.8

MAX_WORKERS      = 3     # proyectos en paralelo -- cada uno hace 2 consultas a Overpass.
                          # sube con cuidado: mas hilos = mas riesgo de rate-limiting.
DELAY_POR_HILO   = 1.5   # pausa que hace CADA hilo entre sus propias solicitudes
GUARDAR_CADA_N   = 10

OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
]
_HEADERS = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'User-Agent': 'TesisViviendaSocialLima/1.0 (uso academico, recoleccion masiva paralela de POIs)',
}

OSM_FILTERS = {
    'Bus':             [('highway','bus_stop'), ('amenity','bus_station')],
    'Primaria':        [('amenity','school')],
    'Universitaria':   [('amenity','university'), ('amenity','college')],
    'Hospitales':      [('amenity','hospital'), ('amenity','clinic')],
    'Alimentacion':    [('amenity','restaurant'), ('amenity','cafe'),
                         ('amenity','fast_food'), ('amenity','food_court')],
    'Tienda':          [('shop','convenience'), ('shop','supermarket')],
    'EspaciosPublicos':[('leisure','park'), ('leisure','playground'), ('place','square')],
    'Farmacias':       [('amenity','pharmacy')],
}
POI_EXTRAER = list(OSM_FILTERS.keys())
COLUMNAS_DETALLE = ['id_proyecto','categoria','poi_nombre','dist_m','poi_lat','poi_lon','dist_caminando_m']


# ═══════════════════════════ FUNCIONES (identicas al notebook) ═══════════════════════════
def clasificar(tags):
    cats = []
    for cat in POI_EXTRAER:
        for k, v in OSM_FILTERS[cat]:
            if tags.get(k) == v:
                cats.append(cat); break
    return cats

def _union_query(lat, lon, radio_m):
    partes = ['[out:json][timeout:90];', '(']
    for cat in POI_EXTRAER:
        for k, v in OSM_FILTERS[cat]:
            partes.append(f'node["{k}"="{v}"](around:{radio_m},{lat},{lon});')
            partes.append(f'way["{k}"="{v}"](around:{radio_m},{lat},{lon});')
    partes.append(');'); partes.append('out center tags;')
    return '\n'.join(partes)

def bbox_alrededor(lat, lon, radio_m):
    dlat = radio_m / 111320
    dlon = radio_m / (111320 * abs(math.cos(math.radians(lat))))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon

def construir_query_red_vial(lat, lon, radio_m):
    s, w, n, e = bbox_alrededor(lat, lon, radio_m)
    return (f'[out:json][timeout:60];'
            f'way["highway"]["highway"!~"motorway|motorway_link|trunk|trunk_link"]["foot"!~"no"]'
            f'({s},{w},{n},{e});out geom;')

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _post_overpass(query, timeout=90):
    body = 'data=' + requests.utils.quote(query)
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data=body, headers=_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(8)
        except Exception:
            continue
    return None

def normalizar_nombre(s):
    if not isinstance(s, str): return ''
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

def nombre_similar(a, b, umbral=UMBRAL_SIMILITUD_NOMBRE):
    if not a or not b: return False
    return SequenceMatcher(None, a, b).ratio() >= umbral

PATRONES_COLEGIO = [r'\binstitucion educativa\b', r'\bi\.?e\.?\b', r'\binst\.?\s*educ\w*\b']
def limpiar_nombre_colegio(nombre_norm):
    n = nombre_norm
    for p in PATRONES_COLEGIO:
        n = re.sub(p, '', n)
    return re.sub(r'\s+', ' ', n).strip()

def deduplicar_por_nombre(grupo, col_nombre, umbral=UMBRAL_SIMILITUD_NOMBRE):
    grupo = grupo.sort_values('dist_caminando_m')
    aceptados, filas_ok = [], []
    for _, row in grupo.iterrows():
        if any(nombre_similar(row[col_nombre], n, umbral) for n in aceptados):
            continue
        aceptados.append(row[col_nombre]); filas_ok.append(row)
    return pd.DataFrame(filas_ok) if filas_ok else grupo.iloc[0:0]


def procesar_proyecto(pid, lat, lon):
    """Devuelve un DataFrame con los POIs YA filtrados, o None si fallo Overpass."""
    data_pois = _post_overpass(_union_query(lat, lon, RADIO_M))
    if data_pois is None:
        return None

    filas = []
    for el in data_pois.get('elements', []):
        tags = el.get('tags', {}) or {}
        cats = clasificar(tags)
        if not cats: continue
        if el.get('type') == 'node':
            plat, plon = el.get('lat'), el.get('lon')
        else:
            c = el.get('center') or {}
            plat, plon = c.get('lat'), c.get('lon')
        if plat is None or plon is None: continue
        dist_m = round(haversine_m(lat, lon, plat, plon), 1)
        nombre = tags.get('name', '')
        for cat in cats:
            filas.append({'categoria': cat, 'poi_nombre': nombre, 'dist_m': dist_m,
                          'poi_lat': plat, 'poi_lon': plon})

    df_pois = pd.DataFrame(filas)
    if df_pois.empty:
        return pd.DataFrame(columns=COLUMNAS_DETALLE)
    df_pois.insert(0, 'id_proyecto', pid)

    data_red = _post_overpass(construir_query_red_vial(lat, lon, BBOX_RADIO_RED_VIAL))
    if data_red is None:
        df_pois['dist_caminando_m'] = None
    else:
        ways = [el for el in data_red.get('elements', []) if el.get('type')=='way' and el.get('geometry') and len(el['geometry'])>1]
        if not ways:
            df_pois['dist_caminando_m'] = None
        else:
            def node_key(pt): return f"{pt['lat']:.5f},{pt['lon']:.5f}"
            nodos, adj = {}, {}
            for w in ways:
                geom = w['geometry']
                for i in range(len(geom)-1):
                    a, b = geom[i], geom[i+1]
                    ka, kb = node_key(a), node_key(b)
                    nodos[ka] = (a['lat'], a['lon']); nodos[kb] = (b['lat'], b['lon'])
                    d = haversine_m(a['lat'],a['lon'], b['lat'],b['lon'])
                    adj.setdefault(ka, []).append((kb, d))
                    adj.setdefault(kb, []).append((ka, d))

            node_keys = list(nodos.keys())
            node_coords = np.array([nodos[k] for k in node_keys])

            def proyectar(plat, plon):
                x = (plon - lon) * 111320 * np.cos(np.radians(lat))
                y = (plat - lat) * 111320
                return x, y

            nx, ny = proyectar(node_coords[:,0], node_coords[:,1])
            arbol = cKDTree(np.column_stack([nx, ny]))

            px0, py0 = proyectar(np.array([lat]), np.array([lon]))
            _, idx0 = arbol.query(np.column_stack([px0, py0]))
            start_key = node_keys[idx0[0]]

            dist_desde_proyecto = {start_key: 0.0}
            pq = [(0.0, start_key)]; visitados = set()
            while pq:
                d, u = heapq.heappop(pq)
                if u in visitados: continue
                visitados.add(u)
                for v, w2 in adj.get(u, []):
                    nd = d + w2
                    if nd < dist_desde_proyecto.get(v, float('inf')):
                        dist_desde_proyecto[v] = nd
                        heapq.heappush(pq, (nd, v))

            poi_x, poi_y = proyectar(df_pois['poi_lat'].values, df_pois['poi_lon'].values)
            dist_snap, idx_snap = arbol.query(np.column_stack([poi_x, poi_y]))
            resultados = []
            for d_snap, idx in zip(dist_snap, idx_snap):
                resultados.append(None if d_snap > UMBRAL_SNAP_M else dist_desde_proyecto.get(node_keys[idx]))
            df_pois['dist_caminando_m'] = [round(r,1) if r is not None else None for r in resultados]

    df_pois = df_pois[df_pois['dist_caminando_m'].notna() & (df_pois['dist_caminando_m'] <= LIMITE_DIST_CAMINANDO_M)].copy()
    df_pois = df_pois[df_pois['poi_nombre'].astype(str).str.strip() != ''].copy()
    if df_pois.empty:
        return df_pois
    df_pois['nombre_norm'] = df_pois['poi_nombre'].map(normalizar_nombre)

    grupos_filtrados = []
    for cat, grupo in df_pois.groupby('categoria'):
        grupo = grupo.copy()
        if cat == 'Primaria':
            grupo['nombre_comparar'] = grupo['nombre_norm'].map(limpiar_nombre_colegio)
            grupo = deduplicar_por_nombre(grupo, 'nombre_comparar')
        elif cat == 'Universitaria':
            grupo = grupo[~grupo['nombre_norm'].str.contains('instituto', na=False)]
        elif cat == 'Tienda':
            conteo = grupo['nombre_norm'].value_counts()
            repetidos = set(conteo[conteo > 1].index)
            es_repetido = grupo['nombre_norm'].isin(repetidos)
            grupo = grupo[(~es_repetido) | (es_repetido & (grupo['dist_caminando_m'] > 50))]
        grupos_filtrados.append(grupo)

    resultado = pd.concat(grupos_filtrados, ignore_index=True) if grupos_filtrados else df_pois.iloc[0:0]
    return resultado.drop(columns=['nombre_comparar'], errors='ignore')


# ═══════════════════════════ CARGA DE PROYECTOS ═══════════════════════════
def cargar_proyectos():
    df = pd.read_excel(RUTA_PROYECTOS)
    print(f'Proyectos en el archivo: {len(df)}')
    df = df[df['precision_geocoding'].isin(['direccion','calle'])].copy()
    print(f'Con direccion real: {len(df)}')
    antes = len(df)
    df = df.drop_duplicates(subset=['LATITUD','LONGITUD'], keep='first').reset_index(drop=True)
    print(f'Tras quitar coordenadas repetidas: {len(df)}  (-{antes-len(df)})')
    return df


# ═══════════════════════════ MAIN (paralelo) ═══════════════════════════
_lock = threading.Lock()

def worker(pid, lat, lon):
    resultado = procesar_proyecto(pid, lat, lon)
    time.sleep(DELAY_POR_HILO)
    return pid, resultado

def main():
    proyectos = cargar_proyectos()

    if os.path.exists(RUTA_IDS_OK):
        ids_ok = set(pd.read_csv(RUTA_IDS_OK)['id_proyecto'].tolist())
        print(f'Ya procesados (no se tocan): {len(ids_ok)}')
    else:
        ids_ok = set()

    if os.path.exists(RUTA_DETALLE):
        detalle_acumulado = pd.read_csv(RUTA_DETALLE)
    else:
        detalle_acumulado = pd.DataFrame(columns=COLUMNAS_DETALLE)

    pendientes = proyectos[~proyectos['id_proyecto'].isin(ids_ok)].reset_index(drop=True)
    print(f'Pendientes: {len(pendientes)} de {len(proyectos)}')
    if len(pendientes) == 0:
        print('Nada pendiente.')
    else:
        t0 = time.time()
        completados = 0
        nuevas_filas = []

        def guardar():
            nonlocal detalle_acumulado, nuevas_filas
            if nuevas_filas:
                detalle_acumulado = pd.concat([detalle_acumulado] + nuevas_filas, ignore_index=True)
                nuevas_filas = []
            detalle_acumulado.to_csv(RUTA_DETALLE, index=False)
            pd.DataFrame({'id_proyecto': sorted(ids_ok)}).to_csv(RUTA_IDS_OK, index=False)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futuros = {
                executor.submit(worker, row['id_proyecto'], row['LATITUD'], row['LONGITUD']): row['id_proyecto']
                for _, row in pendientes.iterrows()
            }
            for futuro in as_completed(futuros):
                pid, resultado = futuro.result()
                completados += 1
                if resultado is None:
                    estado = 'FALLO (Overpass)'
                else:
                    with _lock:
                        nuevas_filas.append(resultado)
                        ids_ok.add(pid)
                    estado = f'ok ({len(resultado)} POIs tras filtros)'

                elapsed = time.time() - t0
                vel = completados / elapsed if elapsed > 0 else 0
                eta_min = (len(pendientes) - completados) / vel / 60 if vel > 0 else 0
                print(f'[{completados}/{len(pendientes)}] id {pid}: {estado}  |  '
                      f'{vel:.2f} proy/s  |  ETA ~{eta_min:.1f} min')

                if completados % GUARDAR_CADA_N == 0:
                    with _lock:
                        guardar()
                    print(f'  --- checkpoint guardado ({len(detalle_acumulado)+sum(len(f) for f in nuevas_filas)} filas, {len(ids_ok)} proyectos ok) ---')

        with _lock:
            guardar()
        print(f'\nTerminado en {(time.time()-t0)/60:.1f} min. Total ok: {len(ids_ok)} de {len(proyectos)}')

    # ── generar archivo de conteos ──
    detalle_final = pd.read_csv(RUTA_DETALLE)
    print(f'\nFilas de detalle: {len(detalle_final)}')
    if 'id_proyecto' not in detalle_final.columns or detalle_final.empty:
        print('Sin filas validas -- no se genera el archivo de conteos.')
        conteos = pd.DataFrame(columns=['id_proyecto'] + POI_EXTRAER)
    else:
        conteos = (detalle_final.groupby(['id_proyecto','categoria']).size()
                                .unstack('categoria', fill_value=0))
        for c in POI_EXTRAER:
            if c not in conteos.columns: conteos[c] = 0
        conteos = conteos[POI_EXTRAER].reset_index()
    conteos.to_csv(RUTA_CONTEOS, index=False)
    print(f'Exportado: {RUTA_CONTEOS}  ({len(conteos)} proyectos)')


if __name__ == '__main__':
    main()
