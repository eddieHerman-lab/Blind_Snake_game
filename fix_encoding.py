# fix_encoding2.py
with open('main.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# corrige os caracteres corrompidos mais comuns
import re

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

fixes = {
    'Ã§': 'ç', 'Ã£': 'ã', 'Ã©': 'é', 'Ã¡': 'á',
    'Ã³': 'ó', 'Ã ': 'à', 'Ãª': 'ê', 'Ã­': 'í',
    'Ãµ': 'õ', 'Ã´': 'ô', 'Ã¢': 'â', 'Ãº': 'ú',
    'Ã‡': 'Ç', 'Ã"': 'Ó', 'Ã‰': 'É', 'Ã€': 'À'
}

for errado, certo in fixes.items():
    content = content.replace(errado, certo)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Corrigido!")

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Corrigido!")
