#%% good
class VtSingleton(type):
    """
    单例，应用方式:静态变量 __metaclass__ = Singleton
    """    
    _instances = {}
    #----------------------------------------------------------------------
    def __call__(cls, *args, **kwargs):
        """调用"""
        if cls not in cls._instances:
            cls._instances[cls] = super(VtSingleton, cls).__call__(*args, **kwargs)
            
        return cls._instances[cls]

class mySing1(object,metaclass = VtSingleton):
#    __metaclass__ = VtSingleton
    def __init__(self):
        self.name = 'myname'
    def myprint(self):
        print('tsfasf')
mm = mySing1()
nn = mySing1()

print(id(mm))
print(id(nn))        
#%%    
class SingletonMeta(type):
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls, *args, **kwargs)
        else:
            print("instance already existed!")
        return cls.instance



class mySing2(object,metaclass = SingletonMeta):
    __metaclass__ = SingletonMeta
    def __init__(self):
        self.name = 'myname'
    def myprint(self):
        print('tsfasf')  

mm = mySing2()
nn = mySing2()

print(id(mm))
print(id(nn))         
#%% good
class Singleton(object):
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls, *args, **kwargs)
        return cls.instance

t1 = Singleton()
t2 = Singleton()
print(id(t1),'\n',id(t2))

#%% good
class Singleton2(type):  
    def __init__(cls, name, bases, dict):  
        super(Singleton2, cls).__init__(name, bases, dict)  
        cls._instance = None  
    def __call__(cls, *args, **kw):  
        if cls._instance is None:  
            cls._instance = super(Singleton2, cls).__call__(*args, **kw)  
        return cls._instance  
  
class MyClass3(object,metaclass = Singleton2): 
    pass
  
one = MyClass3()  
two = MyClass3()  
print(id(one))
print(id(two))

#%% 
class Basic(type):
  def __new__(cls, name, bases, newattrs):
    print( "new: %r %r %r %r" % (cls, name, bases, newattrs))
    return super(Basic, cls).__new__(cls, name, bases, newattrs)
 
  def __call__(self, *args):
    print( "call: %r %r" % (self, args))
    return super(Basic, self).__call__(*args)
 
  def __init__(cls, name, bases, newattrs):
    print( "init: %r %r %r %r" % (cls, name, bases, newattrs))
    super(Basic, cls).__init__(name, bases, dict)
 
 
class Foo(metaclass = Basic):
  __metaclass__ = Basic
 
  def __init__(self, *args, **kw):
    print( "init: %r %r %r" % (self, args, kw))
# 第二次实例化Foo时，先调用的是__call__，因此在该函数中进行singleton设计 
a = Foo('b')
print(id(a))
b = Foo('b')
print(id(b))
