import numpy as np
from PIL import Image



def load_image(path):
    
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[...,:3]


def save_image(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    img = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def dodge(A, B):
    
    A_clamped = np.clip(A, 0.0, 1.0)
    B_clamped = np.clip(B, 0.0, 1.0)

    C = np.empty_like(A_clamped)
    for row in range(A_clamped.shape[0]):
        for column in range(A_clamped.shape[1]):
            for color in range(3):
                a = A_clamped[row,column,color]
                b = B_clamped[row,column,color]
                if a >=1.0 - 1e-6:
                    C[row,column,color]=1.0
                    continue
                C[row,column,color]=b/(1.0-a)
    
    return np.clip(C, 0.0, 1.0)


def burn(A, B):
    
    eps = 1e-6
    A_clamped = np.clip(A, 0.0, 1.0)
    B_clamped = np.clip(B, 0.0, 1.0)

    C = np.empty_like(A_clamped)
    for row in range(A_clamped.shape[0]):
        for column in range(A_clamped.shape[1]):
            for color in range(3):
                a = A_clamped[row,column,color]
                b = B_clamped[row,column,color]
                if a<=1e-6:
                    C[row,column,color]=0.0
                    continue
                C[row,column,color]=1.0-((1.0-b)/a)
    return np.clip(C, 0.0, 1.0)


def darken(A, B):

    A_clamped = np.clip(A, 0.0, 1.0)
    B_clamped = np.clip(B, 0.0, 1.0)
    C = np.empty_like(A_clamped)
    for row in range(A_clamped.shape[0]):
        for column in range(A_clamped.shape[1]):
            for color in range(3):
                a = A_clamped[row,column,color]
                b = B_clamped[row,column,color]
                C[row,column,color] = min(a,b)
    return np.clip(C, 0.0, 1.0)

def difference(A, B):
    
    A_clamped = np.clip(A, 0.0, 1.0)
    B_clamped = np.clip(B, 0.0, 1.0)

    C = np.empty_like(A_clamped)
    for row in range(A_clamped.shape[0]):
        for column in range(A_clamped.shape[1]):
            for color in range(3):
                a = A_clamped[row,column,color]
                b = B_clamped[row,column,color]
                C[row,column,color] = abs(a-b)

    return np.clip(C,0.0,1.0)

def main(img_a_path, img_b_path):

    A = load_image(img_a_path)
    B = load_image(img_b_path)
    
    if A.shape != B.shape:
        raise ValueError("Images A and B must have the same size and mode")

    C_dodge = dodge(A, B)
    C_burn = burn(A, B)
    C_darken = darken(A, B)
    C_diff = difference(A, B)

    save_image(C_dodge, "result_dodge.png")
    save_image(C_burn, "result_burn.png")
    save_image(C_darken, "result_darken.png")
    save_image(C_diff, "result_difference.png")


if __name__ == "__main__":
    img_a_path = "A.jpg"
    img_b_path = "B.jpg"
    main(img_a_path,img_b_path)
