import math
pi = math.pi
def cart_to_polar(x,y):
    r = math.sqrt( x ** 2 + y ** 2)
    if x > 0 and y > 0:
        theta = math.atan(y / x)
    elif (y > 0 and x < 0) or (y < 0 and x < 0):
        theta = math.atan(y / x) + math.pi
    elif x > 0 and y < 0:
        theta = math.atan(y / x) + 2 * math.pi
    elif y == 0 and x < 0:
        theta = math.pi
    elif y == 0 and x > 0:
        theta = 0
        
    print (r, theta)
    
def polar_to_cart(r, theta):
    cart = (r * math.cos(theta), r * math.sin(theta))
    print (cart)


def is_on_uc(x, y):
    d = math.sqrt((x ** 2) + (y ** 2))
    if d == 1:
        print('The point (', x, ',', y, ') is on the Unit Circle')
    else:
        print ('The point (', x, y, ') is NOT on the Unit Circle')

def adv_loc_los(A, a, b):
    math.sin(A) / 