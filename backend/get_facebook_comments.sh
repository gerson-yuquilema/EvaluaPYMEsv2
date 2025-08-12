#!/bin/bash

# Variables
API_URL="http://localhost:8000/api/facebook-scrapper"
COMPANY_NAME="$1"
PAGE_ID="$2"

if [ -z "$COMPANY_NAME" ] || [ -z "$PAGE_ID" ]; then
  echo "Uso: $0 <nombre_empresa> <id_pagina_facebook>"
  exit 1
fi

# Llama a la API y guarda el resultado en un archivo
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"companyName\": \"$COMPANY_NAME\", \"facebookPageId\": \"$PAGE_ID\"}" \
  -o "${COMPANY_NAME}_comentarios.json"

echo "Comentarios guardados en ${COMPANY_NAME}_comentarios.json"chmod +x scripts/get_facebook_comments.sh