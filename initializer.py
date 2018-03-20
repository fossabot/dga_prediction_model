import config
from modules.get_training_data import get_data
from modules.get_training_data import format_data
from modules.training_model import train_model


# Interactive command-line menu
def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("""
    1. Get training data
    2. Train model
    3. Testing
    4. Start capture network traffic
    5. Exit
    """)
    print(67 * "-")


def remark():
    print("""
    Remark:
    If you are setting up algorithm for the first time, follow the steps 1-4.
    When you restart, you can use the fourth step without initializing the 
    learning process again.
    """)
    print(67 * "-")


def main():
    loop = True
    remark_count = 1

    while loop:
        print_menu()
        if remark_count == 1:
            remark()
            remark_count -= 1
        choice = input("Enter your choice [1-5]: ")
        print(67 * "-")
        if choice == "1":
            print("Getting training data...")
            get_data()
            print("Reduction to a unified form...")
            format_data()
            input("Press Enter to continue...")

        elif choice == "2":
            print("Training model... (it may take a few minutes)")
            train_model()
            input("Press Enter to continue...")

        elif choice == "3":
            print("Starting capture network traffic")
            input("Press Enter to continue...")

        elif choice == "4":
            print("Exiting")
            loop = False
        else:
            print("Wrong option selection. Enter any key to try again..")


if __name__ == "__main__":
    main()
