songFile = open("music-files-for-data-generation/1.txt")
X_dataFile = open("data/X_train.csv", "w")
Y_dataFile = open("data/Y_train.csv", "w")
dataFile = open("data/data.csv", "w")
totalSongs = 15
data_point_list = ['A', 'A', 'A', 'A']
notes_in_one_data_point = 4
X_bytes_per_line = 107


'''
00000000000000000000000000
AaBbCcDdEeFfGgZz12486"/#.-
'''
note_representation_dict = {"A":0,"a":1,"B":2,"b":3,"C":4,"c":5,"D":6,"d":7,"E":8,"e":9,
                            "F":10,"f":11,"G":12,"g":13,"Z":14,"z":15,"#":16,".":17,
                            "-":18,"/":19,"1":20,'"2':21,"4":22,"8":23,"16":24,"\"":25}

def getNote():
    currNote = ['0'] * 26;
    chr = songFile.read(1);
    while (chr != " "):
        if chr == ';':
            return ';'
        if chr == '1':
            temp = songFile.read(1)
            if temp >= '0' and temp <= '6':
                num = int(chr + temp)
                num = bin(num)
                digits = [x for x in str(num)]
                del digits[:2]
                digits.reverse()
                currNote[note_representation_dict['1']:(note_representation_dict['1'] + len(digits))] = digits
                chr = songFile.read(1)
            elif temp != " ":
                currNote[note_representation_dict[chr]] = '1'
                currNote[note_representation_dict[temp]] = '1'
                chr = songFile.read(1)
            else:
                currNote[note_representation_dict[chr]] = '1'
                break
            continue
        if chr >= '2' and chr <= '9':
            num = int(chr)
            num = bin(num)
            digits = [x for x in str(num)]
            del digits[:2]
            digits.reverse()
            currNote[note_representation_dict['1']:(note_representation_dict['1'] + len(digits))] = digits
            chr = songFile.read(1)
            continue
        currNote[note_representation_dict[chr]] = '1'
        chr = songFile.read(1)
    return "".join(currNote)
    
def getData():
    data_point_list[2], data_point_list[1], data_point_list[0] = data_point_list[3], data_point_list[2], data_point_list[1]
    data_point_list[3] = getNote()
    return data_point_list[3]

# create X_train :   
for fileIndex in range(1,totalSongs + 1):
    print("dealing with file : "+str(fileIndex))
    songFile = open("music-files-for-data-generation/"+str(fileIndex)+".txt")
    getData()
    getData()
    getData()
    while(getData() != ";"):
        for i in range(notes_in_one_data_point):
            X_dataFile.write(data_point_list[i])
            if i == (notes_in_one_data_point - 1):
                continue;
            X_dataFile.write(",")
        X_dataFile.write("\n")
    X_dataFile.seek(X_dataFile.tell() - X_bytes_per_line)
    songFile.close()
        
        
# create Y_train :
for fileIndex in range(1,totalSongs + 1):
    print("dealing with file : "+str(fileIndex))
    songFile = open("music-files-for-data-generation/"+str(fileIndex)+".txt")
    getData()
    getData()
    getData()
    getData()
    y = getData();
    while(y != ";"):
        Y_dataFile.write(y)
        Y_dataFile.write("\n")
        y = getData()
    songFile.close()

    
data_point_list = ['A','A','A','A','A']
def getData_2():
    data_point_list[3], data_point_list[2], data_point_list[1], data_point_list[0] = data_point_list[4], data_point_list[3], data_point_list[2], data_point_list[1]
    data_point_list[4] = getNote()
    return data_point_list[4]
    
# create data.py :
notes_in_one_data_point = len(data_point_list) 
for fileIndex in range(1,totalSongs + 1):
    print("dealing with file : "+str(fileIndex))
    songFile = open("music-files-for-data-generation/"+str(fileIndex)+".txt")
    getData_2()
    getData_2()
    getData_2()
    getData_2()
    while(getData_2() != ";"):
        for i in range(notes_in_one_data_point):
            dataFile.write(data_point_list[i])
            if i == (notes_in_one_data_point - 1):
                continue;
            dataFile.write(",")
        dataFile.write("\n")
    songFile.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    