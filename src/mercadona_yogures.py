"""
PAIBS – Precios de yogures de Mercadona (proyecto académico, ESADE)

Qué hace, en sencillo:
  1. Pide a la web de Mercadona la lista de categorías.
  2. Se queda con las que contienen "yogur" en el nombre.
  3. Entra en cada una y apunta los productos (nombre, precio, tamaño, precio/kg).
  4. Entra en cada producto para sacar su código de barras (EAN) y su marca.
  5. Lo guarda todo en precios_mercadona_yogures.csv

Cómo ejecutarlo:
  - En Colab: sube este archivo y ejecuta  !python mercadona_yogures.py
  - En VS Code: abre una terminal y ejecuta  python mercadona_yogures.py
  Necesita: pip install requests pandas

Notas:
  - Usa la API interna de la web de Mercadona, que no es oficial: puede cambiar sin aviso.
  - Va despacio a propósito (pausa entre peticiones) para no cargar sus servidores.
  - Los precios cambian según la zona (almacén). WAREHOUSE elige cuál; "mad1" = Madrid.
"""

import json
import time
from datetime import date

import pandas as pd
import requests

BASE = "https://tienda.mercadona.es/api"
WAREHOUSE = "mad1"          # prueba "bcn1" si queréis precios de la zona de Barcelona
PARAMS = {"lang": "es", "wh": WAREHOUSE}
PAUSA_SEG = 1.5             # pausa entre peticiones: no bajar de 1 segundo
HEADERS = {"User-Agent": "PAIBS-ESADE student project (uso academico)"}
PALABRA = "yogur"           # categorías que buscamos
SALIDA = "precios_mercadona_yogures.csv"


def pedir(ruta):
    """Descarga un JSON de la API, con pausa y con un reintento si falla."""
    url = f"{BASE}{ruta}"
    for intento in range(2):
        time.sleep(PAUSA_SEG)
        r = requests.get(url, params=PARAMS, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  ⚠️ {r.status_code} en {url} (intento {intento + 1})")
    return None


def categorias_de_yogur():
    """Recorre el árbol de categorías y devuelve las que contienen 'yogur'."""
    datos = pedir("/categories/")
    if not datos:
        raise SystemExit("No se pudo leer la lista de categorías. ¿Ha cambiado la API?")
    encontradas = []
    for grupo in datos.get("results", []):
        for sub in grupo.get("categories", []):
            if PALABRA in sub.get("name", "").lower():
                encontradas.append((sub["id"], sub["name"]))
    return encontradas


def productos_de_categoria(cat_id):
    """Devuelve los productos de una categoría (pueden estar en subsecciones)."""
    datos = pedir(f"/categories/{cat_id}/")
    if not datos:
        return []
    productos = []
    for seccion in datos.get("categories", []):
        for p in seccion.get("products", []):
            p["_seccion"] = seccion.get("name")
            productos.append(p)
    return productos


def main():
    cats = categorias_de_yogur()
    print(f"Categorías con '{PALABRA}': {[n for _, n in cats]}")

    filas = []
    for cat_id, cat_nombre in cats:
        prods = productos_de_categoria(cat_id)
        print(f"\n{cat_nombre}: {len(prods)} productos")
        for p in prods:
            pi = p.get("price_instructions", {}) or {}
            detalle = pedir(f"/products/{p['id']}/") or {}
            filas.append({
                "barcode": detalle.get("ean"),
                "mercadona_id": p.get("id"),
                "nombre": p.get("display_name"),
                "marca": detalle.get("brand"),
                "formato": p.get("packaging"),
                "precio_eur": pi.get("unit_price"),
                "tamano": pi.get("unit_size"),
                "unidad_tamano": pi.get("size_format"),
                "precio_referencia_eur": pi.get("reference_price") or pi.get("bulk_price"),
                "unidad_referencia": pi.get("reference_format"),
                "categoria": cat_nombre,
                "seccion": p.get("_seccion"),
                "tienda": "Mercadona",
                "almacen": WAREHOUSE,
                "fecha": date.today().isoformat(),
                "url": p.get("share_url"),
            })
            print(f"  ✓ {p.get('display_name')} – {pi.get('unit_price')} €")

    df = pd.DataFrame(filas)
    df.to_csv(SALIDA, index=False)
    print(f"\nGuardado {len(df)} productos en {SALIDA}")
    print(f"Con código de barras: {df['barcode'].notna().sum()}")
    print(f"Marcas: {df['marca'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
