When this program is run, you will be prompted to choose from the 2 datasets available. Enter 1 or 2 to choose between the dataset with 131 cities or 237 cities.

The next prompt will ask which method you would like to choose. Enter 1, 2 or 3 to choose between the Book implementation, Tournament selection implementation and Leveled Mentorship implementation (my novel approach to the problem)

After this selection you will be asked weather you want to use the default values set by the program. Choose 1 for the default values and 2 to input your own.
The default values are
POP SIZE = 100
CROSSOVER PROPORTION = 0.7
MUTATION RATE = 0.2
NUMBER OF GENERATIONS = 100
LEVEL PROPORTION SPLIT = (0.6, 0.3, 0.1)

If you choose to use your own values you will be prompted to input them now. Enter values according to the prompt and separate the values using a comma. Whitespaces will be deleted so the program should work regardles of spaces before or after commas.

The Genetic Algorithm will then create the initial population of 1024 individuals. This may take a few seconds to create.

Finally, if the Leveled Mentorship Implementation was chosen, you will be prompted to input the level proportion split.
Input the values according to the prompt separated by commas.

The model will then begin training and a progress percentage will be shown. When the number of generations are reached, the model will output the individual with the best fitness and the path it took.

This implementation always starts and ends with the first city given by the dataset.