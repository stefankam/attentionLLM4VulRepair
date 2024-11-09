#print all the paths from root node that sums up to the given sum
#This function return all the paths from given node which it takes as its argument in the form of list of strings
#This function uses level order traversal
def f(root,s):
    if not root:
        return ['None']
    <vul/>if root.val==s:
        return [str(root.val) + ' ']</vul>
    left=f(root.left,s-root.val)
    right=f(root.right,s-root.val)

    new=[]
    for e in left+right:
        new.append(str(root.val)+ ' ' + e)

    return new


class node:
    def __init__(self,val):
        self.val=val

    left,right,s=None,None,0


'''

'''
#This the tree which we formed below

         10
       /    \
     28      13
           /    \
         14      15
        /   \    / \
       21   22  23  24




#Driver code
root=node(10)
root.left=node(28)
root.right=node(13)
root.right.left=node(14)
root.right.right=node(15)
root.right.left.left=node(21)
root.right.left.right=node(22)
root.right.right.left=node(23)
root.right.right.right=node(24)

for string in f(root,38):
    if 'None' not in string:
        print(string)
