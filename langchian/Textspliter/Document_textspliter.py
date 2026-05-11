from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # function inside class
    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")

    def is_adult(self):
        if self.age >= 18:
            return True
        return False


# creating object
user1 = User("May", 26)

# calling functions
user1.display_info()

print("Adult:", user1.is_adult())"""

spliter =RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 400,
    chunk_overlap = 0,
)

chunk = spliter.split_text(text)

print(chunk[0])


