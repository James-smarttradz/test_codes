# refer here https://www.geeksforgeeks.org/command-line-arguments-in-python/
import getopt, sys

# python test_getopt.py --Output=hello
# python test_getopt.py -o hello
# python test_getopt.py -h -m
# python test_getopt.py -h this_h -m this_m -o hello

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

# Options
options = "hmo:"

# Long options
long_options = ["Help", "My_file", "Output ="]

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
    print('Arguments:', arguments)
    print('Values:', values)

    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-h", "--Help"):
            print ("Diplaying Help")

        elif currentArgument in ("-m", "--My_file"):
            print ("Displaying file_name:", sys.argv[0])

        elif currentArgument in ("-o", "--Output"):
            print (("Enabling special output mode (% s)") % (currentValue))

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
