import math

# entropy of salary
print('Calculating entropy for Salary (high/low = yes/no = true/false):')
entropy = (-(0.4 * math.log(0.4, 2)) - (0.6 * math.log(0.6, 2)))
print(entropy, '\n')

# entropy of Education level
print('Calculating Entropy for Salary on Education Level:')
Eeducation = (5/10) * (-(1/5 * math.log(1/5, 2)) - (4/5 * math.log(4/5, 2))) + \
    (5/10 * (-(3/5 * math.log(3/5, 2)) - (2/5 * math.log(2/5, 2))))
print(Eeducation)

# Gain (Salary, Ed. Level)
print('Calculating Gain for Education Level:')
Geducation = entropy - Eeducation
print(Geducation, '\n')

# Entropy of Years of Experience
print('Calculating Entropy for Salary on years of experience:')
Eyears = (3/10) * (-(1/3 * math.log(1/3, 2)) - (2/3 * math.log(2/3, 2))) + \
    (4/10 * (-(2/4 * math.log(2/4, 2)) - (2/4 * math.log(2/4, 2)))) + \
    (3/10) * (-(1/3 * math.log(1/3, 2)) - (2/3 * math.log(2/3, 2)))
print(Eyears)

# Gain (salary, years of experience)
print('Calculating Gain for years of experience:')
print(entropy-Eyears, '\n')

# Gain of Career
print('Calculating Entropy for Salary on Career:')
Ecareer = (5/10) * (-(1/5 * math.log(1/5, 2)) - (4/5 * math.log(4/5, 2))) + \
    (5/10 * (-(3/5 * math.log(3/5, 2)) - (2/5 * math.log(2/5, 2))))
print(Ecareer)
print('Calculating Gain for Career:')
print(entropy - Ecareer, '\n')

print('------------------------------------------------')
print('Choice: Education level,\t Gain: ', Geducation)
Ehs = (-(1/5 * math.log(1/5, 2)) - (4/5 * math.log(4/5, 2)))
# high school path
print('\t Choice: High School, Entropy: ', Ehs)
Ecollege = (-(3/5 * math.log(3/5, 2)) - (2/5 * math.log(2/5, 2)))
Ec = (3/5 * (-(1/3 * math.log(1/3, 2)) - (2/3 * math.log(2/3, 2)))) + \
    (2/5 * (-(2/2 * math.log(2/2, 2)) - 0))
print('\t\t Choice: Career, Entropy: ', Ec)
print('\t\t Gain on Career: ', Ehs - Ec)
print('\t\t LESS GAIN NOT USED, FOLLOW Years of Experience')
print('\t\t------------------------------------------------')
y = (0) + (0) + (2/5 * (-(1/2 * math.log(1/2, 2)) - (1/2 * math.log(1/2, 2))))
print('\t\t Choice: Years of Experience, Entropy: ', y)
print('\t\t Gain on Years of Experience: ',  Ehs - y)
print('\t\t\t Less than 3, Entropy: 0.0')
print('\t\t\t 3 to 10, Entropy: 0.0')
ye = (-(1/2 * math.log(1/2, 2)) - (1/2 * math.log(1/2, 2)))
print('\t\t\t More than 10, Entropy: ', ye)
print('\t\t\t\t Choice: Career, Entropy: 1.0')
print('\t\t\t\t Gain on Career: 1.0')
print('\t\t\t\t\t Management, Entropy: 0.0')
print('\t\t\t\t\t Services, Entropy: 0.0')
# college path
print('\t------------------------------------------------')
print('\t Choice: College, Entropy: ', Ecollege)
Cc = (3/5 * (-(1/3 * math.log(1/3, 2)) - (2/3 * math.log(2/3, 2)))) + \
    (2/5 * (-(2/2 * math.log(2/2, 2)) - 0))
print('\t\t Choice: Career, Entropy: ', Cc)
print('\t\t Gain on Career: ', Ecollege - Cc)
print('\t\t\t Management, Entropy: 0.0')
# (1/3 * (0) + (1/3 * (0) + 1/3*(0)))
ecS = (-(1/3 * math.log(1/3, 2)) - (2/3 * math.log(2/3, 2)))
print('\t\t\t Services, Entropy: ', ecS)
ecsY = (1/3 * (-(1/1 * math.log(1/1, 2)) - 0)) + (1/3 *
                                                  (-(1/1 * math.log(1/1, 2)) - (0))) + (1/3 * (-(0) - (1/1 * math.log(1/1, 2))))
print('\t\t\t\t Choice: Years of Experience, Entropy: ', ecsY)
print('\t\t\t\t Gain on Years of Experience: ', ecS)
print('\t\t\t\t\t Less than 3: 0.0')
print('\t\t\t\t\t 3 to 10: 0.0')
print('\t\t\t\t\t More than 10: 0.0')
print('\t\t------------------------------------------------')
Cy = (2/5 * (-(1/2 * math.log(1/2, 2)) - (1/2 * math.log(1/2, 2)))) + (2/5 * (-(1/2 *
                                                                                math.log(1/2, 2)) - (1/2 * math.log(1/2, 2)))) + (1/5 * (-(1/1 * math.log(1/1, 2)) - 0))
print('\t\t Choice: Years of Experience, Entropy: ', Cy)
print('\t\t Gain on Years of Experience: ',  Ecollege - Cy)
print('\t\t LESS GAIN NOT USED, FOLLOW Career')

Cye = (2/5 * (-(1/2 * math.log(1/2, 2)) - (1/2 * math.log(1/2, 2)))) + \
    (2/5 * (-(1/2 * math.log(1/2, 2)) - (1/2 * math.log(1/2, 2)))) + \
    (1/5 * (-(1/1 * math.log(1/1, 2)) - (0)))
