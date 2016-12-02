import GA
import numpy as np
import matplotlib.pyplot as plt

GArun = GA.GA()
print('Evolution in progress...')
start = GArun.pop.get_fittest().fitness

for i in range(100):
    GArun.evolve()
    name = 'figs/result_' + str(i) + '.png'
    GArun.pop.get_fittest().save_plot(name)

end = GArun.pop.get_fittest().fitness  
print('Evolution done! Final score is: ' + str(round(end)) + '. That is ' + str(round((1 - end / start) * 100)) 
       + ' % better than the best generation 1 solution.')

GArun.pop.get_fittest().plot()