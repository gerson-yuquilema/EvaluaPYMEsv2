from facebook_scraper import get_posts

def obtener_comentarios(post_id):
    comentarios = []
    for post in get_posts(post_urls=[f'https://www.facebook.com/{post_id}'], options={"comments": True}):
        for comentario in post.get('comments_full', []):
            comentarios.append({
                'usuario': comentario['commenter_name'],
                'texto': comentario['comment_text'],
                'fecha': comentario['comment_time']
            })
    return comentarios

# Ejemplo de uso
post_id = '1234567890'  # Extraído de la URL del formulario
comentarios = obtener_comentarios(post_id)
print(comentarios)