from fastapi import FastAPI, Request, HTTPException
from supabase import create_client, Client
import os

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Faltan variables de entorno")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/")
def root():
    return {"message": "API de Scoring para PYMEs activa"}

@app.post("/api/guardar-entrada")
async def guardar_entrada(request: Request):
    body = await request.json()
    company_name = body.get("companyName", "empresa")
    facebook_url = body.get("facebookUrl", "")
    facebook_page_id = body.get("facebookPageId", "")
    twitter_profile = body.get("twitterProfile", "")
    tweet_id = body.get("tweetId", "")

    txt_content = (
        f"Empresa: {company_name}\n"
        f"Facebook URL: {facebook_url}\n"
        f"Facebook Page ID: {facebook_page_id}\n"
        f"Twitter Profile: {twitter_profile}\n"
        f"Tweet ID: {tweet_id}\n"
    )

    bucket = "comentarios"
    carpeta = company_name.replace(" ", "_")
    filename = f"{carpeta}/entrada.txt"

    try:
        supabase.storage.from_(bucket).upload(filename, txt_content.encode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo en Supabase: {str(e)}")

    return {"success": True, "detail": "Archivo de entrada guardado correctamente"}