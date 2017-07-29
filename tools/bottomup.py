from collections import deque
data_dir = '../data/deps_sample'
fhand = open(data_dir)
sents = []
sent = []
for line in fhand:
    line = line.strip()
    if line == '...EOS...':
        if not sent==[]:
            sents.append(sent)
            sent = [] # reset 
        continue
    else:
        sent.append(line)
fhand.close()

queues = []
for sent in sents:
    actions = []
    queue = []
    for line in sent:
        queue.append(tuple([int(line.split()[0]), int(line.split()[3])]))
    queue.reverse()
    queues.append(queue)

for queue in queues:
    stack = []
    actions = []
    FIN = False
    stack.append(queue.pop())
    stack.append(queue.pop())
    while not FIN: 
        if len(stack) == 1 and len(queue) == 0:
            FIN = True
        if len(stack) <= 1:
            stack.append(queue.pop())
        elif stack[-2][0] == stack[-1][1]:
            stack.pop()
            remain = stack.pop()
            stack.append(remain)
        elif stack[-2][1] == stack[-1][0]:
            remain = stack.pop()
            stack.pop()
            stack.append(remain)
        else:
            stack.append(queue.pop())
        print(stack)
        print(queue)

        

