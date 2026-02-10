import sys
from modes.register import run_register
from modes.recognize import run_recognize

def main():
    print("=" * 50)
    print("Sistema de Reconocimiento Facial")
    print("=" * 50)
    print("1. Modo Registro")
    print("2. Modo Reconocimiento")
    print("3. Salir")
    print("=" * 50)
    
    choice = input("Seleccione una opción: ").strip()
    
    if choice == "1":
        run_register()
    elif choice == "2":
        run_recognize()
    elif choice == "3":
        print("Saliendo...")
        sys.exit(0)
    else:
        print("Opción inválida")

if __name__ == "__main__":
    main()