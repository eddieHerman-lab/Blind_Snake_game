# fix_encoding2.py
with open('main.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# corrige os caracteres corrompidos mais comuns
content = content.replace('Ã§', 'ç')
content = content.replace('Ã£', 'ã')
content = content.replace('Ã©', 'é')
content = content.replace('Ã¡', 'á')
content = content.replace('Ã³', 'ó')
content = content.replace('Ã ', 'à')
content = content.replace('Ãª', 'ê')
content = content.replace('Ã­', 'í')
content = content.replace('Ãµ', 'õ')
content = content.replace('Ã´', 'ô')
content = content.replace('Ã¢', 'â')
content = content.replace('Ãº', 'ú')

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Corrigido!")
