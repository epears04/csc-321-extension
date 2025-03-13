import torch
import string
from util.model import LSTMModel

model = LSTMModel(93, 16, 32, 1, 5)
model.load_state_dict(torch.load('output/password_model.pth'))
model.eval()

chars = string.ascii_letters + string.digits + r"!@#$%^&*()/'\:~.<>?{}[]=+_-;|`"
char_to_num = {char: i + 1 for i, char in enumerate(chars)}
char_to_num["<UNK>"] = 0 # Use index 0 for unknown characters

def encode_password(password, max_len=16):
    encoded = [char_to_num.get(char, 0) for char in password]
    encoded = encoded[:max_len]  # Truncate if too long
    encoded += [0] * (max_len - len(encoded))  # Pad if too short
    return torch.tensor([encoded], dtype=torch.float32)  # Convert to tensor

def predict_strenght(password):
    encoded_password = encode_password(password)
    with torch.no_grad():
        output = model(encoded_password)
        _, predicted = torch.max(output, 1)

    strengths = ["Very Weak", "Weak", "Average", "Strong", "Very Strong"]
    return strengths[predicted.item()]

#CLI Loop
if __name__ == '__main__':
    print("AI-Powered Password Evaluator")
    print("How to Use:")
    print("1. When prompted please input a password you want to test.")
    print("2. The program will take the password and run it through our model.")
    print("3. A category will return ranging from 'Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong' signifying how strong your password is.")
    print("4. You can test as many passwords as you like.")
    print("5. To exit the program enter 'q' when ready.")
    print("Please be aware that this AI is not 100% accurate.")
    print("Please follow NIST Password Guidelines or a random password generator to ensure the best security possible.\n")


    while True:
        password = input("Enter Your Password: ")
        if password.lower() == "q":
            print("Thank you for using this AI-Powered Password Evaluator")
            break
        strength = predict_strenght(password)
        print("Your password is: {}".format(strength))