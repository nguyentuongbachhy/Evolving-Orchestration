from typing import Dict, List, Tuple
import random

class TestCases:
    
    @staticmethod
    def get_all_test_cases() -> Dict[str, List[Tuple[str, str]]]:
        """Get all test cases organized by type"""
        return {
            "math": TestCases.get_math_test_cases(),
            "research": TestCases.get_research_test_cases(),
            "code": TestCases.get_code_test_cases(),
            "mixed": TestCases.get_mixed_test_cases()
        }
    
    @staticmethod
    def get_math_test_cases() -> List[Tuple[str, str]]:
        """Mathematical calculation test cases"""
        return [
            ("What is 2**10 + 3 - 2**9?", "515"),
            ("Calculate 15 * 8 + 47", "167"),
            ("What is the square root of 144?", "12"),
            ("Solve: 25 * 4 - 18", "82"),
            ("What is 7**3 + 5**2?", "368"),
            ("Calculate (100 - 25) / 5", "15"),
            ("What is 3.14 * 10**2?", "314"),
            ("Solve: 2**8 - 2**6", "192"),
            ("What is 9! / 8!?", "9"),
            ("Calculate 45 + 67 - 23", "89"),
            ("What is 12**2 + 13**2?", "313"),
            ("Solve: 1000 / 25 + 15", "55"),
            ("What is 6 * 7 * 8?", "336"),
            ("Calculate 2**16", "65536"),
            ("What is 99 + 1 - 50?", "50"),
            ("What is (50 * 4) / (5 + 5)?", "20"),
            ("Solve: 2**5 * 3**3", "864"),
            ("Calculate factorial of 6 (6!)", "720"),
            ("What is sqrt(1024)?", "32"),
            ("Solve: (200 - 50) * (10 + 2)", "1800"),
            ("What is 15**3 - 14**3?", "919"),
            ("Calculate (7! / 5!) + 100", "940"),
            ("Solve: (81**0.5) + (64**0.5)", "17"),
            ("What is 2**10 + 3**6?", "1853"),
            ("Calculate (12345 % 100) + (500 / 25)", "70"),
            ("What is (1001 * 1001) - (1000 * 1000)?", "2001"),
            ("Solve: 9**4 / 3**6", "9"),
            ("What is (8! / 6!) + (5**2)?", "526"),
            ("Calculate (2**20) / (2**10)", "1024"),
            ("Solve: 3**7 - 2**10", "1859"),
            ("What is (45 * 11) - (99 * 5)?", "0"),
            ("Calculate (1000000 / 1000) + (999 * 2)", "2999"),
            ("Solve: (14**2 + 48**2)**0.5", "50"),
            ("What is (10! / 8!) - (7 * 6)", "504"),
            ("Calculate (29**2 - 21**2)", "464"),
        ]
    
    @staticmethod
    def get_research_test_cases() -> List[Tuple[str, str]]:
        """Research and information gathering test cases"""
        return [
            ("What is the capital of France?", "Paris"),
            ("Who invented the telephone?", "Alexander Graham Bell"),
            ("What year was Python programming language created?", "1991"),
            ("What is the largest planet in our solar system?", "Jupiter"),
            ("Who wrote the novel '1984'?", "George Orwell"),
            ("What is the chemical symbol for gold?", "Au"),
            ("In what year did World War II end?", "1945"),
            ("What is the fastest land animal?", "Cheetah"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
            ("What is the smallest country in the world?", "Vatican City"),
            ("What element has the atomic number 1?", "Hydrogen"),
            ("In what year was the Internet invented?", "1969"),
            ("What is the hardest natural substance?", "Diamond"),
            ("Who developed the theory of relativity?", "Albert Einstein"),
            ("What is the currency of Japan?", "Yen"),
            ("Who was the first person to travel into space?", "Yuri Gagarin"),
            ("What is the longest river in the world?", "Nile"),
            ("Who is known as the Father of Computers?", "Charles Babbage"),
            ("What is the capital of Australia?", "Canberra"),
            ("Which element has the chemical symbol 'Na'?", "Sodium"),
            ("Who was the first woman to win a Nobel Prize?", "Marie Curie"),
            ("What is the tallest mountain on Earth?", "Mount Everest"),
            ("In what year did the Berlin Wall fall?", "1989"),
            ("Who discovered penicillin?", "Alexander Fleming"),
            ("What is the national language of Brazil?", "Portuguese"),
            ("What is the unit of electrical resistance?", "Ohm"),
            ("Who was the first President of the United States?", "George Washington"),
            ("Which scientist proposed the three laws of motion?", "Isaac Newton"),
            ("What is the capital of Canada?", "Ottawa"),
            ("In what year did the Titanic sink?", "1912"),
            ("Which desert is the largest in the world?", "Sahara"),
            ("Who composed the opera 'The Magic Flute'?", "Wolfgang Amadeus Mozart"),
            ("What is the currency of South Korea?", "Won"),
            ("Which planet is known as the Red Planet?", "Mars"),
            ("Who is the author of 'The Odyssey'?", "Homer")
        ]
    
    @staticmethod
    def get_code_test_cases() -> List[Tuple[str, str]]:
        """Programming and code execution test cases"""
        return [
            ("Write a Python function to calculate factorial of 5", "120"),
            ("Create a list comprehension to get even numbers from 1 to 10", "[2, 4, 6, 8, 10]"),
            ("Write code to reverse the string 'hello'", "olleh"),
            ("Calculate the sum of numbers from 1 to 100", "5050"),
            ("Write a function to check if 17 is prime", "True"),
            ("Create a dictionary with keys a, b, c and values 1, 2, 3", "{'a': 1, 'b': 2, 'c': 3}"),
            ("Write code to find the maximum number in [3, 7, 2, 9, 1]", "9"),
            ("Calculate fibonacci number at position 10", "55"),
            ("Write code to count vowels in 'programming'", "3"),
            ("Create a lambda function to square a number and apply to 5", "25")
        ]
    
    @staticmethod
    def get_mixed_test_cases() -> List[Tuple[str, str]]:
        """Multi-step tasks requiring multiple agents"""
        return [
            ("Research the year Python was created and calculate 2024 minus that year", "33"),
            ("Find the atomic number of carbon and multiply by 3", "18"),
            ("Look up the speed of light and convert to km/s", "299792.458"),
            ("Research Einstein's birth year and calculate his age if alive in 2024", "145"),
            ("Find the number of planets in solar system and calculate 2**planets", "256"),
            ("Research when iPhone was released and add 15 years", "2022"),
            ("Look up the height of Mount Everest in meters and divide by 1000", "8.848"),
            ("Find the boiling point of water in Celsius and multiply by 2", "200"),
            ("Research the number of continents and calculate factorial", "5040"),
            ("Look up the year WWI ended and add 100", "2018"),
            ("Implement a stack using Python list and push elements [1,2,3], then pop once", "[1, 2]"),
            ("Write a function to perform binary search on [1,3,5,7,9] for 7", "3"),
            ("Create a binary tree with root=10, left=5, right=15 and return inorder traversal", "[5, 10, 15]"),
            ("Implement BFS traversal on graph {0:[1,2],1:[2],2:[3],3:[]} starting from 0", "[0, 1, 2, 3]"),
            ("Sort the list [9,1,4,7,3] using quicksort", "[1, 3, 4, 7, 9]"),
            ("Use dynamic programming to compute nth Fibonacci where n=20", "6765"),
            ("Calculate sigmoid(0) in ML", "0.5"),
            ("Perform one step of gradient descent with w=0.5, x=2, y=1, lr=0.1 for linear regression", "0.6"),
            ("Create a NumPy array [1,2,3] and compute dot product with [4,5,6]", "32"),
            ("Using pandas, create DataFrame with column A=[1,2,3] and return mean of A", "2.0")
        ]
    
    @staticmethod
    def get_training_set(problem_type: str, train_ratio: float = 0.8) -> Dict[str, List[Tuple[str, str]]]:
        """Split test cases into training and test sets"""
        all_cases = TestCases.get_all_test_cases()
        
        if problem_type not in all_cases:
            return {"train": [], "test": []}
        
        cases = all_cases[problem_type].copy()
        random.shuffle(cases)
        
        split_point = int(len(cases) * train_ratio)
        
        return {
            "train": cases[:split_point],
            "test": cases[split_point:]
        }
    
    @staticmethod
    def get_sample_tasks(num_samples: int = 5) -> List[Tuple[str, str]]:
        """Get random sample of tasks for quick testing"""
        all_cases = TestCases.get_all_test_cases()
        all_tasks = []
        
        for _, tasks in all_cases.items():
            all_tasks.extend(tasks)
        
        return random.sample(all_tasks, min(num_samples, len(all_tasks)))