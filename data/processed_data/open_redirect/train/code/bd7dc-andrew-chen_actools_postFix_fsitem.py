"""
	Use
		file:///Users/chenan/Other/python-2.7.2-docs-html/library/os.html#files-and-directories
	and
		file:///Users/chenan/Other/python-2.7.2-docs-html/library/os.path.html

	but make a more OO interface to dealing with files and folders.

	Envisioned class hierarchy would be something like:

	FSItem
		Folder
		File

	and there would be intuitive properties/methods
	for FSItem, Folder, and File
"""

import os
import os.path
import fnmatch

def specialized(p):
	if os.path.isdir(p):
		return Folder(p)
	elif os.path.isfile(p):
		return File(p)
	elif os.path.islink(p):
		return Link(p)
	else:
		return FSItem(p)

def fs_object(p):
	p = os.path.abspath(p)
	return specialized(p)

class FSPath(object):
	def __init__(self,path):
		self.path = path
	def __sub__(self,other):
		if (self.path.startswith(other.path)):
			r = self.path[len(other.path):]
			assert((other.path+r) == self.path)
			return r[1:]
		else:
			raise IndexError, (other.path,self.path)
	
	# from os.path
	def abspath(self):
		return FSPath(os.path.abspath(self.path))
	def basename(self):
		return os.path.basename(self.path)
	def dirname(self):
		return FSPath(os.path.dirname(self.path))
	def exists(self):
		return FSPath(os.path.exists(self.path))
	def lexists(self):
		return FSPath(os.path.lexists(self.path))
	def expanduser(self):
		return FSPath(os.path.expanduser(self.path))
	def expandvars(self):
		return FSPath(os.path.expandvars(self.path))
	def getatime(self):
		return os.path.getatime(self.path)
	def getctime(self):
		return os.path.getctime(self.path)
	def getsize(self):
		return os.path.getsize(self.path)
	def isabs(self):
		return os.path.isabs(self.path)
	def isfile(self):
		return os.path.isfile(self.path)
	def isdir(self):
		return os.path.isdir(self.path)
	def islink(self):
		return os.path.islink(self.path)
	def ismount(self):
		return os.path.ismount(self.path)
	def normcase(self):
		return FSPath(os.path.normcase(self.path))
	def realpath(self):
		return FSPath(os.path.realpath(self.path))
	def relpath(self,start=None):
		if start is None:
			return FSPath(os.path.relpath(self.path))
		else:
			return FSPath(os.path.relpath(self.path,start))
	def samefile(self,other):
		if isinstance(other,FSPath):
			return os.path.samefile(self.path,other.path)
		else:
			return os.path.samefile(self.path,other)
	def split(self):
		head,tail = os.path.split(self.path)
		return (FSPath(head),FSPath(tail))
	def splitdrive(self):
		return os.path.splitdrive(self.path)
	def splitext(self):
		return os.path.splitext(self.path)
	def splitunc(self):
		return os.path.splitunc(self.path)
	
	# from os
	def access(self,mode):
		return os.access(self.path,mode)
	def chdir(self):
		return os.chdir(self.path)
		
	@staticmethod
	def getcwd(self):
		return FSPath(os.getcwd())
	
	def chflags(self,flags):
		return os.chflags(self.path,flags)
	def chroot(self):
		return os.chroot(self.path)
	def chmod(self,mode):
		return os.chmod(self.path,mode)
	def chown(self,uid=-1,gid=-1):
		return os.chown(self.path,uid,gid)
	def lchflags(self,flags):
		return os.lchflags(self.path,flags)
	def lchmod(self,mode):
		return os.lchmod(self.path,mode)
	def lchown(self,uid=-1,gid=-1):
		return os.lchown(self.path,uid,gid)
	def link(self,link_name):
		return os.link(self.path,link_name)
	def listdir(self):
		# need to wrap this better in a subclass
		return os.listdir(self.path)
	def lstat(self):
		return os.lstat(self.path)
	def mkfifo(self,mode=None):
		if mode is None:
			return os.mkfifo(self.path)
		else:
			return os.mkfifo(self.path,mode)
	# mknod, major, minor, makedev not implemented
	def mkdir(self,mode=None):
		if mode is None:
			return os.mkdir(self.path)
		else:
			return os.mkdir(self.path,mode)
	def makedirs(self,mode=None):
		if mode is None:
			return os.makedirs(self.path)
		else:
			return os.makedirs(self.path,mode)
	# pathconf, pathconf_names not implemented
	def readlink(self):
		return FSPath(os.readlink(self.path))
	def remove(self):
		return os.remove(self.path)
	def removedirs(self):
		return os.removedirs(self.path)
	def rename(self,other):
		if isinstance(other,FSPath):
			return os.rename(self.path,other.path)
		else:
			return os.rename(self.path,other)
	def renames(self,other):
		if isinstance(other,FSPath):
			return os.renames(self.path,other.path)
		else:
			return os.renames(self.path,other)
	def rmdir(self):
		return os.rmdir(self.path)
	def stat(self):
		return os.stat(self.path)
	def statvfs(self):
		return os.statvfs(self.path)
	def symlink(self,link_name):
		return os.symlink(self.path,link_name)
	def unlink(self):
		return os.unlink(self.path)
	def utime(self,times):
		return os.utime(self.path,times)
	def walk(self,topdown=True, onerror=None, followlinks=False):
		return os.walk(self.path,topdown,onerror,followlinks)


class FSPathList(list):
	def _raw(self):
		return map(lambda x: x.path,self)
	
	# from os.path
	def commonprefix(self):
		return FSPath(os.path.commonprefix(self._raw()))
	def join(self):
		return FSPath(*(self._raw()))

class FSItem(FSPath):
	def __init__(self,path):
		super(FSItem,self).__init__(path)
		assert(self.isabs())
		assert(self.lexists())
	def parent(self):
		p = self.dirname()
		return Folder(p.path)
	def ancestors(self):
		c = FSItem(self.path)
		p = c.parent()
		while c.path is not p.path:
			yield p
			c = p
			p = c.parent()
		
	def common_parent(self,other):
		l = FSPathList()
		l.append(self)
		l.append(other)
		r = l.commonprefix()
		if r.isdir():
			return Folder(r.path)
		else:
			return Folder(r.dirname().path)
	def walk(self,*args,**kwargs):
		r = super(FSItem,self).walk(*args,**kwargs)
		for root, dirs, files in r:
			folder_list = [Folder(os.path.join(root,dir)) for dir in dirs]
			file_list = [File(os.path.join(root,a_file)) for a_file in files]
			yield (Folder(root),folder_list,file_list)

class Folder(FSItem):
	def __init__(self,path):
		super(Folder,self).__init__(path)
		assert(self.isdir())
	def __iter__(self):
		r = self.items()
		return r
	def items(self):
		for item in self.listdir():
			p = os.path.abspath(os.path.join(self.path,item))
			yield specialized(p)
	def __getitem__(self,name):
		for item in self.items():
			if item.basename() == name:
				return item
		if name == "":
			return self
		raise IndexError, (name,self.path)
	def folders(self):
		for item in self.listdir():
			p = os.path.abspath(os.path.join(self.path,item))
			if os.path.isdir(p):
				yield Folder(p)
	def files(self):
		for item in self.listdir():
			p = os.path.abspath(os.path.join(self.path,item))
			if os.path.isfile(p):
				yield File(p)
	
	def create(self,what,name):
		t = os.path.join(self.path,name)
		if what is Folder:
			os.mkdir(t)
			return Folder(t)
		elif what is File:
			f = open(t,"w")
			f.close()
			return File(t)
		else:
			raise NotImplementedError
	
	def file_with_name(self,name):
		try:
			the_file = self[name]
		except IndexError:
			the_file = self.create(File,name)
		return the_file

	def folder_with_name(self,name):
		try:
			the_folder = self[name]
		except IndexError:
			the_folder = self.create(Folder,name)
		return the_folder

def safe_open(path_or_File,mode):
	if isinstance(path_or_File,File):
		return path_or_File.open(mode)
	else:
		return open(path_or_File,mode)

class Line(object):
	<fix/>def __init__(self,file,number,text,column=0,original=None):</fix>
		self.file = file
		self.number = number
		self.text = text
		self.column = column
		if original is None:
			self.original = self
		else:
			self.original = original
	def clone(self,file=None,number=None,text=None,column=None,original=None,leading_whitespace=None):
		result = Line(self.file,self.number,self.text,self.column,self.original)
		if file is None: pass
		else: result.file = file
		if number is None: pass
		else: result.number = number
		if text is None: pass
		else: result.text = text
		if column is None: pass
		else: result.column = column
		if original is None: pass
		else: result.original = original
		
		if leading_whitespace is None: pass
		else
			try:
				result.leading_whitespace = self.leading_whitespace
			except:
				pass
			
		return result
		
	def strip_from(self,text):
		l,m,r = self.text.rpartition(text)
		result = self.clone()
		if len(m):
			result.text = l
		else:
			result.text = r
		return result
	def strip(self):
		new_text = self.text.lstrip()
		leading_whitespace = self.text[0:len(self.original.text) - len(new_text)]
		column = len(leading_whitespace)
		result = self.clone(
			text = new_text.rstrip(),
			column = column,
			leading_whitespace = leading_whitespace
		)
		return result
	def __len__(self):
		return len(self.text)
	def split(self):
		return self.text.split()

class File(FSItem):
	def __init__(self,path):
		super(File,self).__init__(path)
		assert(self.isfile())
	def read(self):
		f = open(self.path,"rU")
		r = f.read()
		f.close()
		return r
	def readlines(self):
		f = open(self.path,"rU")
		r = f.readlines()
		f.close()
		return r
	def read_Lines(self):	
		with open(self.path,"rU") as f:
			count = 0
			for l in f:
				count += 1
				yield Line(self,count,l)
	def write(self,o):
		f = open(self.path,"wb")
		r = f.write(o)
		f.close()
		return r
	def writelines(self,o):
		f = open(self.path,"wb")
		r = f.writelines(o)
		f.close()
		return r
	def open(self,mode):
		return open(self.path,mode)
		

class Link(FSItem):
	def __init__(self,path):
		super(Link,self).__init__(path)
		assert(self.islink())

def current():
	r = Folder(os.getcwd())
	return r

def root():
	return Folder("/")

def home():
	return Folder(os.path.expanduser("~"))

def FSItemList(FSPathList):
	def common_parent(self):
		r = None
		for item in self:
			if r is None:
				r = item
			else:
				r = r.common_parent(item)
		return r
	def filter(self,pattern):
		return fnmatch.filter(self,pattern)

if __name__ == "__main__":
	print "current has "
	for item in current():
		print item.path
	print "root has "
	for item in root():
		print item.path
	print "home has "
	for item in home():
		print item.path
