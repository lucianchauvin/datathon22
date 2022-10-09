# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
import numpy as np
from glob import glob
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import random
import itertools
import math

# Import helper functions from utils.py
import utils

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`555
    """

    def __init__(self):
        self.possibleCombos = set(itertools.permutations([0, 1, 2, 3]))
        self.possibleCombos = [([x[0], x[1]], [x[2], x[3]]) for x in self.possibleCombos]

    def returnAvrg(self, lst, n=22): #THIS IS THE PROBLEM
        rav = sum([x[0] for x in lst])/len(lst)
        gav = sum([x[1] for x in lst])/len(lst)
        bav = sum([x[2] for x in lst])/len(lst)

        rlstSqr = sum([(x[0] - rav)**2 for x in lst])/len(lst)
        glstSqr = sum([(x[1] - gav)**2 for x in lst])/len(lst)
        blstSqr = sum([(x[2] - bav)**2 for x in lst])/len(lst)

        n = 22

        if rlstSqr > n and glstSqr > n and blstSqr > n:
            

            p1r = sum(x[0] for x in lst[:len(lst)//3])/(len(lst)//3)
            p2r = sum(x[0] for x in lst[len(lst)//3:(len(lst)//3)*2])/(len(lst)//3)
            p3r = sum(x[0] for x in lst[(len(lst)//3)*2:(len(lst)//3)*3])/(len(lst)//3)

            p1g = sum(x[1] for x in lst[:len(lst)//3])/(len(lst)//3)
            p2g = sum(x[1] for x in lst[len(lst)//3:(len(lst)//3)*2])/(len(lst)//3)
            p3g = sum(x[1] for x in lst[(len(lst)//3)*2:(len(lst)//3)*3])/(len(lst)//3)

            p1b = sum(x[2] for x in lst[:len(lst)//3])/(len(lst)//3)
            p2b = sum(x[2] for x in lst[len(lst)//3:(len(lst)//3)*2])/(len(lst)//3)
            p3b = sum(x[2]
                    for x in lst[(len(lst)//3)*2:(len(lst)//3)*3])/(len(lst)//3)

            p1 = (p1r, p1g, p1b)
            p2 = (p2r, p2g, p2b)
            p3 = (p3r, p3g, p3b)
            # p1 = (p1r + p2r + p3r)/3
            # p2 = (p1g + p2g + p3g)/3
            # p3 = (p1b + p2b + p3b)/3

            out = [p1, p2, p3]
            
            # print(out)
            return out
        else:
            # print("SADGE")
            return [(random.randint(999,9999999), random.randint(999,9999999), random.randint(999,9999999)), (random.randint(999,9999999), random.randint(999,9999999), random.randint(999,9999999)), (random.randint(999,9999999), random.randint(999,9999999), random.randint(999,9999999))]


    def generateStates(self, img):
        l = [0,2,1,3]
        
        img

        width, height = img.size
        stateWidth = width/2
        stateHeight = height/2

        states = {}

        c = 0
        for (i, j) in [(i, j) for i in range(2) for j in range(2)]:
            top = j*stateWidth
            left = i*stateHeight
            bottom = top + stateWidth
            right = left + stateWidth

            s = img.crop((left, top, right, bottom))
            # s.show()

            pixels = np.array(s)

            adj = [self.returnAvrg(pixels[0].tolist()), self.returnAvrg([pixels[y][len(pixels[0])-1].tolist()
                                                                         for y in range(len(pixels))]), self.returnAvrg(pixels[len(pixels)-1].tolist()), self.returnAvrg([pixels[y][0].tolist() for y in range(len(pixels))])]

            states.update(
                {str(l[c]): {"adj": adj}})

            c += 1

        # with open("states.json", "w") as f:
        #     json.dump(states, f)

        return states

    def solve(self, states):
        pass
   

            
    def make_prediction(self, img_path):
        img = Image.open(f'{img_path}')

        s = self.generateStates(img)
        
        # gross = sum([abs(myStates[k][z] - otherStates[directions[k]][z]) for z in range(len(otherStates[k]))])
        solutionsWithPerdic = []
        directions = [2,3,0,1]
        for i in self.possibleCombos:
            score = 0
            score += sum([sum([abs(s[str(i[0][0])]["adj"][1][z][p] - s[str(i[0][1])]["adj"][3][z][p]) for p in range(3)]) for z in range(len(s[str(i[0][0])]["adj"][0]))])

            score += sum([sum([abs(s[str(i[0][0])]["adj"][2][z][p] - s[str(i[1][0])]["adj"][0][z][p]) for p in range(3)]) for z in range(len(s[str(i[0][0])]["adj"][0]))])

            score += sum([sum([abs(s[str(i[0][1])]["adj"][2][z][p] - s[str(i[1][1])]["adj"][0][z][p]) for p in range(3)]) for z in range(len(s[str(i[0][0])]["adj"][0]))])

            score += sum([sum([abs(s[str(i[1][0])]["adj"][1][z][p] - s[str(i[1][1])]["adj"][3][z][p]) for p in range(3)]) for z in range(len(s[str(i[0][1])]["adj"][0]))])

            solutionsWithPerdic.append((i, score))



            solutionsWithPerdic.append([i,score])
        solutionsWithPerdic.sort(key=lambda x: x[1])
        best = solutionsWithPerdic[0][0]
        # print(solutionsWithPerdic)
        b = "".join([str(x) for x in best[0]]) + \
            "".join([str(x) for x in best[1]])
        return str(b.index("0")) + str(b.index("1")) + str(b.index("2")) + str(b.index("3"))



# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':
    sucess = 0
    total = 0

    for img_name in glob(f'train\\*\\*.png'):
        # Open an example image using the PIL library

        # if random.choice([0,1]) == 0:
        #     continue

        example_image = Image.open(img_name)

        # Use instance of the Predictor class to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)
        if prediction != img_name.split('\\')[1]:
            # print(f'Incorrect prediction for {img_name}: {prediction}')
            pass
        else:
            sucess += 1
        total += 1

    print(f"Sucess: {(sucess/total)*100}")

    # #     # Example images are all shuffled in the "3120" order
       

    # Use instance of the Predictor class to predict the correct order of the current example image
    # predictor = Predictor()
    # # prediction = predictor.make_prediction("train\\3021\\00018.png")
    # prediction = predictor.make_prediction("train\\0231\\00203.png")
    # print(prediction)
