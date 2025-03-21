# Estrutura do Addon Super Empty Rig

Este documento explica a estrutura de arquivos necessária para o addon funcionar corretamente.

## Estrutura de Diretórios

A estrutura correta do addon deve ser:

```
super_empty/
├── __init__.py
└── templates/
    └── super_empty_template.blend
```

## Instalação Manual

Para instalar o addon manualmente:

1. Certifique-se de que a estrutura de arquivos esteja correta como mostrado acima
2. Compacte a pasta `super_empty` em um arquivo ZIP
3. No Blender, vá para Edit > Preferences > Add-ons
4. Clique em "Install..." e selecione o arquivo ZIP criado
5. Ative o addon marcando a caixa ao lado de "Rigging: Super Empty Rig"

## Usando o Script de Empacotamento

O script `package.py` incluído neste projeto criará automaticamente um arquivo ZIP com a estrutura correta. Para usá-lo:

1. Certifique-se de que todos os arquivos estejam nos lugares corretos:
   - `super_empty/__init__.py`
   - `super_empty/templates/super_empty_template.blend`

2. Execute o script:
   ```
   python package.py
   ```

3. O script criará um arquivo ZIP com o nome `Super_Empty_Rig_1.0.0.zip` (ou similar, dependendo da versão)

4. Use este arquivo ZIP para instalar o addon no Blender

## Observações Importantes

- O arquivo de template (`super_empty_template.blend`) deve estar na pasta `templates` dentro da pasta principal do addon
- O código do addon assume que o arquivo de template está neste local específico
- Não renomeie os arquivos ou pastas, pois isso pode quebrar o funcionamento do addon 