
import traceback

x = "hello"

#if condition returns True, then nothing happens:
assert x == "hello"

#if condition returns False, AssertionError is raised:
try:
    assert x == "goodbye", "x should be 'hello'"

except AssertionError as error:
    # print('Assertion error')
    print(error)
    # traceback.print_exc()
    # or save into variable
    error_message = traceback.format_exc()
finally:
    print(error_message)
    
