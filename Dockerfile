# Usa una imagen base con Python y Node preinstalados
FROM python:3.10-slim

# Instala Node.js
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Crea directorio de trabajo
WORKDIR /app

# Copia todos los archivos del proyecto
COPY . .

# Instala dependencias Python
RUN pip install --no-cache-dir lightgbm transformers sentence-transformers scikit-learn pandas numpy joblib

# Instala dependencias Node
RUN npm install

# Da permisos de ejecución al script
RUN chmod +x start.sh

# Expone los puertos del backend y frontend
EXPOSE 8000
EXPOSE 8001

# Define variables de entorno (puedes sobreescribirlas en tiempo de ejecución)
ENV SUPABASE_URL="https://ywlyxdgcgwdwkyaelvtx.supabase.co"
ENV SUPABASE_KEY="sb_secret_MqJDBd13EXLDKlfacGB2_Q_nx6vJnw_"
ENV BROWSER="echo"  

# Comando de inicio
CMD ["./start.sh"]