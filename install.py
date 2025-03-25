import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

def run_blender_command(blender_path, python_script):
    """Executa um comando Python no Blender"""
    try:
        subprocess.run([blender_path, "--background", "--python", python_script], check=True)
    except subprocess.CalledProcessError:
        print("Aviso: Erro ao executar comando no Blender, mas continuando...")

def create_disable_script():
    """Cria um script temporário para desabilitar o addon"""
    script = """
import bpy
if "super_empty" in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_disable(module="super_empty")
"""
    with open("temp_disable.py", "w") as f:
        f.write(script)

def create_enable_script():
    """Cria um script temporário para habilitar o addon"""
    script = """
import bpy
if "super_empty" not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module="super_empty")
"""
    with open("temp_enable.py", "w") as f:
        f.write(script)

def install_addon():
    # Configurações
    blender_path = r"C:\Users\HenriqueMiranda\Documents\Blender\blender-4.4.0-candidate - DEV USE\blender.exe"
    addons_path = r"C:\Users\HenriqueMiranda\Documents\Blender\blender-4.4.0-candidate - DEV USE\portable\scripts\addons"
    addon_name = "super_empty"
    addon_path = os.path.join(addons_path, addon_name)
    
    # Verifica se o Blender está em execução e fecha
    try:
        subprocess.run(["taskkill", "/F", "/IM", "blender.exe"], check=False)
        time.sleep(2)  # Aguarda o processo ser encerrado
    except:
        print("Nenhuma instância do Blender estava em execução.")
    
    # Desabilita o addon
    print("Desabilitando o addon...")
    create_disable_script()
    run_blender_command(blender_path, "temp_disable.py")
    
    # Remove a pasta do addon se existir
    if os.path.exists(addon_path):
        print(f"Removendo pasta antiga do addon: {addon_path}")
        shutil.rmtree(addon_path)
    
    # Copia o novo addon
    print("Copiando novo addon...")
    shutil.copytree("super_empty", addon_path)
    
    # Habilita o addon
    print("Habilitando o addon...")
    create_enable_script()
    run_blender_command(blender_path, "temp_enable.py")
    
    # Limpa arquivos temporários
    if os.path.exists("temp_disable.py"):
        os.remove("temp_disable.py")
    if os.path.exists("temp_enable.py"):
        os.remove("temp_enable.py")
    
    # Abre o Blender
    print("Abrindo o Blender...")
    subprocess.Popen([blender_path])
    
    print("Instalação concluída!")

if __name__ == "__main__":
    install_addon() 