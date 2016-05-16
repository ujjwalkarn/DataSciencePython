>>> a = ['a', 'b', 'c', 'd', 'e']
>>> for index, item in enumerate(a): print index, item
...
0 a
1 b
2 c
3 d
4 e




#convert a list to string:

list1 = ['1', '2', '3']
str1 = ''.join(list1)

Or if the list is of integers, convert the elements before joining them.

list1 = [1, 2, 3]
str1 = ''.join(str(e) for e in list1)



#FIND method

str.find(str2, beg=0 end=len(string))

Parameters
str2 -- This specifies the string to be searched.
beg -- This is the starting index, by default its 0.
end -- This is the ending index, by default its equal to the lenght of the string.

Return Value
This method returns index if found and -1 otherwise.

str1 = "this is string example....wow!!!";
str2 = "exam";

print str1.find(str2);
print str1.find(str2, 10);
print str1.find(str2, 40);

#15
#15
#-1





#2D LIST PYTHON

# Creates a list containing 5 lists initialized to 0
Matrix = [[0 for x in range(5)] for x in range(5)] 
You can now add items to the list:

Matrix[0][0] = 1
Matrix[4][0] = 5

print Matrix[0][0] # prints 1
print Matrix[4][0] # prints 5


if you have a simple two-dimensional list like this:

A = [[1,2,3,4],
     [5,6,7,8]]
then you can extract a column like this:

def column(matrix, i):
    return [row[i] for row in matrix]
Extracting the second column (index 1):

>>> column(A, 1)
[2, 6]
Or alternatively, simply:

>>> [row[1] for row in A]
[2, 6]
