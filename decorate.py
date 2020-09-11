#https://www.cnblogs.com/lliuye/p/9143676.html
class TreeNode:
    def __init__(self,val):
        self.val=val
        self.left=left
        self.right=right
class OperationTree:
    def create(self,List):
        root=TreeNode(List[0])
        lens=len(List)
        if lens>=2:
            root.left=self.create(List[1])
        if lens>=3:
            root.right=self.create(List[2])
        return root
    def query(self,root,data):
        if root==None:
            return False
        if root.val==data:
            return True
        elif root.left:
            return self.query(root.left,data)
        elif root.right:
            return self.query(root.right,data)
    def preOrder(self,root):
        if root==None:
            return
        print(root,val , end=' ')
        self.preOrder(root.left)
        self.preOrder(root.right)
    def InOrder(self,root):
        if root==None:
            return
        self.InOrder(root.left)
        print(root.val,end=' ')
        self.InOrder(root.right)
    def BacOrder(self,root):
        if root==None:
            return
        self.BacOrder(root.left)
        self.BacOrder(root.right)
        print(root.val,end=' ')
    def BFS(self,root):
        if root==None:
            return
        queue=[]
        vals=[]
        queue.append(root)
        while queue:
            currentNode=queue.pop(0)
            vals.append(currentNode.val)
            print(currentNode.val,end=' ')
            if currentNode.left:
                queue.append(currentNode.left)
            if currentNode.right:
                queue.append(currentNode.right)
        return vals
    def DFS(self,root):
        if root==None:
            return
        stack=[]
        vals=[]
        stack.append(root)
        while stack:
            currentNode=stack.pop()
            vals.append(currentNode.val)
            print(currentNode.val,end=' ')
            if currentNode.right:
                stack.append(currentNode.right)
            if currentNode.left:
                stack.append(currentNode.left)
        return vals
if __name__ == '__main__':
    List1 = [1,[2,[4,[8],[9]],[5]],[3,[6],[7]]]
    op = OperationTree()
    tree1 = op.create(List1)
    print('先序打印：',end = '')
    op.PreOrder(tree1)
    print("")
    print('中序打印：',end = '')
    op.InOrder(tree1)
    print("")
    print('后序打印：',end = '')
    op.BacOrder(tree1)
    print("")
    print('BFS打印 ：',end = '')
    bfs = op.BFS(tree1)
    #print(bfs)
    print("")
    print('DFS打印 ：',end = '')
    dfs = op.DFS(tree1)
    #print(dfs)
    print("")

#https://blog.csdn.net/weixin_42348333/article/details/80991081
def big_endian(arr,start,end):
    root=start
    while True:
        child=root*2+1
        if child>end:
            break
        if child+1<=end and arr[child]<arr[child+1]:
            child+=1
        if arr[root]<arr[child]:
            arr[root],arr[child]=arr[child],arr[root]
            root=child
        else:
            break
def heap_sort(arr):
    first=len(arr)//2-1
    for start in range(first,-1,-1):
        big_endian(arr,start,len(arr)-1)
    for end in range(len(arr)-1,0,-1):
        arr[0],arr[end]=arr[end],arr[0]
        big_endian(arr,0,end-1)
def main():
    l=[3, 1, 4, 9, 6, 7, 5, 8, 2, 10]
    print(l)
    heap_sort(l)
    print(l)
if __name__ == '__main__':
    main()

#https://www.cnblogs.com/Lin-Yi/p/7309143.html
def merge_sort(li):
    if len(li)==1:
        return li
    mid=len(li)//2
    left=li[:mid]
    right=li[mid:]
    l1=merge_sort(left)
    r1=merge_sort(right)
    return merge(l1,r1)
def merge(left,right):
    result=[]
    while len(left)>0 and len(right)>0:
        if left[0]<=right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result += left
    result += right
    return result
if __name__ == '__main__':
    li=[5,4,3,2,1]
    li2=merge_sort(li)
    print(li2)

#https://www.cnblogs.com/kaiping23/p/9614395.html
def quick_sort(li,start,end):
    if start>-end:
        return
    left=start
    right=end
    mid=li[left]
    while left<right:
        while left<right and li[right]>=mid:
            right-=1
        li[left]=li[right]
        while left<right and li[left]<mid:
            left+=1
        li[right]=li[left]
    li[left]=mid
    quick_sort(li,start,left-1)
    quick_sort(li,left+1,end)
if __name__ == '__main__':
    l = [6,5,4,3,2,1]
    quick_sort(l,0,len(l)-1)
    print(l)

graph={
    "A":{"B","C"},
    "B":{"A","C","B"},
    "C":{"A","B","D","E"},
    "D":{"B","C","E","F"},
    "E":{"C","D"},
    "F":{"D"}
}
print(graph.keys())
def BFS(graph,s):
    queue=[]
    queue.append(s)
    seen=set()
    seen.add(s)
    parent={s:None}
    while len(queue)>0:
        vertex=queue.pop(0)
        nodes=graph[vertex]
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
                parent[w]=vertex
            print(vertex)
        return parent
parent=BFS(graph,"E")
print(parent)
print("-------")
for key in parent:
    print(key,parent[key])
v='B'
while v is not None:
    print(v)
    v=parent[v]

graph={
    "A":{"B","C"},
    "B":{"A","C","B"},
    "C":{"A","B","D","E"},
    "D":{"B","C","E","F"},
    "E":{"C","D"},
    "F":{"D"}
}
print(graph.keys())
def DFS(graph,s):
    stack=[]
    stack.append(s)
    seen=set()
    seen.add(s)
    while len(stack)>0:
        vertex=stack.pop()
        nodes=graph[vertex]
        for w in nodes:
            if w not in seen:
                stack.append(w)
                seen.add(w)
        print(vertex)
DFS(graph,"A")

import heapq
import math
graph={
    "A":{"B":5,"C":1},
    "B":{"A":5,"C":2,"D":1},
    "C":{"A":1,"B":2,"D":4,"E":8},
    "D":{"B":1,"C":4,"E":3,"F":6},
    "E":{"C":8,"D":3},
    "F":{"D":3}
}
def init_distance(graph,s):
    distance={s:0}
    for vertex in graph:
        if vertex!=s:
            distance[vertex]=math.inf
    return distance
    
# print(graph["A"].keys())
# # for key in graph:
# #     print(key)
def dijkstra(graph,s):
    pqueue=[]
    heapq.heappush(pqueue,(0,s))
    seen=set()
    parent={s:None}

    distance=init_distance(graph,s)

    while len(pqueue)>0:
        pair=heapq.heappop(pqueue)
        dist=pair[0]
        vertex=pair[1]
        seen.add(vertex)

        nodes=graph[vertex].keys()

        for w in nodes:
            if w not in seen:
                if dist+graph[vertex][w]<distance[w]:
                    heapq.heappush(pqueue,(dist+graph[vertex][w],w))
                    parent[w]=vertex
                    distance[w]=dist+graph[vertex][w]
        print(vertex)
    return parent,distance
parent,distance=dijkstra(graph,"A")
print(parent)
print(dijkstra)

































