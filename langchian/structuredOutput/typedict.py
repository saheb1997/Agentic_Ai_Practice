from typing import TypedDict

class Person (TypedDict):
    name:str
    age:int

newperson :Person={"name":'saheb','age':28}

print(newperson)