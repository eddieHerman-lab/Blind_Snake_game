# converte para utf-8 - roda esse script UMA VEZ na pasta do projeto
with open('main.py', 'r', encoding='cp1252', errors='replace') as f:
    content = f.read()

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Convertido com sucesso!")