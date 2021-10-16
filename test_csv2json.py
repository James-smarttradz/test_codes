from collections import defaultdict

import json
import pandas as pd


# df = pd.read_csv('test.csv')

# json_doc = defaultdict(list)

# _id = 0
# for _id in df.T:
#     # print(_id)
#     data = df.T[_id]
#     # print(data)
#     key = data.course

#     for elt in json_doc[key]:
#         print(elt)
#         if elt["date"] == data.date:
#             elt[data.student] = data.grade
#             break
#     else:
#         values = {'date': data.date, data.student: data.grade}
#         json_doc[key].append(values)

# print(json.dumps(json_doc, indent=4))


#%%
def get_data():
    
    # Opening JSON file 
    # f = open('test_csv2json.json',)
    
    # g = open('input_leadtime.json',) 
      
    # data_json = json.load(f)
    
    df = pd.read_excel('test_csv2json.xlsx',sheet_name='Sheet1')

    return df

def createjson(df):
    
    json_doc = defaultdict(list)
    json_doc2 = defaultdict(list)
    
    for _id in df.T:
        # print(_id)
        data = df.T[_id]
        # print(data)
        key = data.course
        key2 = data.course2
        
        for elt in json_doc[key]:   # level 1
            print(elt)
            # if elt["date"] == data.date:
            #     elt[data.student] = data.grade
            #     break
        
        else:   # The else clause executes after the loop completes normally
            
            for elt2 in json_doc2[key2]:
                print(elt2)
                
            else:
                
                
                values2 = {
                    'name': data['name'], 
                    'age': data['age']
                    }
            json_doc2[key2].append(values2)
            
            values = {
                'type': data.type, 
                'size': data.size, 
                data.course2 : values2
                }
        json_doc[key].append(values)
            
    print(json.dumps(json_doc, indent=4)) #DONE

def main():
    df = get_data()
    
    createjson(df)
    
if __name__ == '__main__':
    main()