import os


from modules.get_training_data import get_data
from modules.get_training_data import format_data
from modules.train_model import train
from modules.capture_domain import capture
import sys


# Interactive command-line menu
def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("""
    1. Get training data
    2. Train model
    3. Start capture network traffic
    4. Exit
    """)
    print(67 * "-")


def remark():
    print("""
    Remark:
    If you are setting up algorithm for the first time,  follow 
    the steps 1-3. When you restart, you can use the third step 
    without initializing the learning process again.
    """)
    print(67 * "-")


def main():
    loop = True
    remark_count = 1
    try:
        while loop:
            print_menu()
            if remark_count == 1:
                remark()
                remark_count -= 1
            choice = input("Enter your choice [1-4]: ")
            print(67 * "-")
            if choice == "1":
                get_data()

                format_data()
                input("Press Enter to continue...")

            elif choice == "2":
                train()
                input("Press Enter to continue...")

            elif choice == "3":
                if os.path.isfile('input data/model.pkl'):
                    capture()
                else:
                    print("You must run the training algoirthm first.")
                input("Press Enter to continue...")

            elif choice == "4":
                print("[*] Exiting...")
                loop = False
            else:
                print("Wrong option selection. Enter any key to try again.")
    except KeyboardInterrupt:
        print("\n[*] Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
