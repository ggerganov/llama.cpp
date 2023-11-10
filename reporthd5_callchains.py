import h5py
import click 
import collections

ids = {}
#with open("string_ids.txt") as fi:
#    for x in fi:
#        p = x.strip().split("|")
#        ids[p[0]] = p[1]
#print(ids)
# from https://stackoverflow.com/a/53340677

def descend_obj(obj,sep='\t', callback=None):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        #print("FILE")
        for key in obj.keys():
            #print ("KEY",sep,'-',key,':',obj[key])
            descend_obj(obj[key],sep=sep+'\t',callback=callback)
    elif type(obj)==h5py._hl.dataset.Dataset:
        #print("ds")
        #print( obj.name, obj.shape, obj.size, obj.dtype)
        return callback(obj)
    else:
        print(obj)

def h5dump(path,group='/', callback=None):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
        print(path)
        descend_obj(f[group],callback=callback)


def get_map(obj):
    global ids
    for x in obj:
        k = x[0]
        v = x[1].decode("utf-8")
        if len(v) >100:
            v = str(v[0:100]).replace("\n","").replace("\t","") +"trunc"
            #print("DEBUG",k,v)
        ids[k] = v
            
def get_data(obj):
    #for x in obj:
    #    print(x[2]
    report = collections.Counter()
    objs = obj.size
    ldepth = 0
    lname = ""
    for i in range(objs):
        #print("OBJ",i, obj[i])
        data = obj[i]
        symbol = data[1]
        pointer = data[4] #instruction pointer
        module = str(data[2]) + ids.get(data[2],"oops") 
        depth = str(data[5])
        idepth = data[5]
            
        name = ids.get(symbol,"oops")
        name = str(name) + "|"+ str(symbol) + "|MOD:" + module + "|DEP:" +depth + "|ORIG:" + str(pointer) +"/" + hex(pointer)
        rname = ""
        if idepth > ldepth:
            rname = lname +"|"+ name            
        else:
            rname = "NEW"+"|"+name


        ldepth = idepth
        lname = name
        #print("\t".join(map(str,data)),name)
        report[rname] += 1
        # 1 [('id', '<i8'), 
        # 2 ('symbol', '<u4'),
        # 3 ('module', '<u4'),
        # 4 ('unresolved', 'u1'),
        # 5 ('originalIP', '<u8'),
        # 6 ('stackDepth', '<i4')]
        #ip = obj[i][4]
        #print("DEB",j,f)
        #    report[ip] += 1
    for k in report.most_common():
        print("\t".join(map(str,k)))
@click.command()
@click.argument("ifile", type=click.Path(exists=True))
def main(ifile):
    #h5dump(ifile,"/")
    h5dump(ifile,"/StringIds",callback=get_map)
    #print(ids)
    h5dump(ifile,"/CUDA_CALLCHAINS",callback=get_data)
if __name__ == "__main__":
    main()
