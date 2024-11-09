from loader import Loader
import openpyxl as pyxl
import os.path
import numpy as np

class Grapher():

	def __init__(self):
		self.loader = Loader()		
		# dictionary of data sets
		self.sets = self.loader.getSets()
		# create a list with all the data sets (that are not blank)
		self.all_sets = []
		for key in self.sets:
			if key != 'blank':
				self.all_sets.append(self.sets[key])

		self.graph_width_param = 4
		self.graph_sub_width_param = 0.5
		self.marker_size = 40
		self.folder_name = 'static/images'
		self.folder_static_name = 'images'

	def findFile(self,file_name):
		#return os.path.isfile(file_name)
		return False
	def expectFileName(self,gene_name,data_set,graph_type,file_type = 'png'):
		file_name = '_'.join([gene_name,data_set.split('_')[0],graph_type])
		file_name = '.'.join([file_name,file_type])
		file_dir = '/'.join([self.folder_name,file_name])
		return file_dir

	def decodeGeneName(self,expected_file):
		file_name = expected_file.split('/')[2]
		gene_name = file_name.split('_')[0]
		print gene_name
		return gene_name

	def decodeDataSet(self,expected_file):
		file_name = expected_file.split('/')[2]
		data_set = file_name.split('_')[1]
		print data_set
		return data_set


	def new_bar_plot(self,gene_name,cell_type,datasets = 'ALL'):
		data = self.loader.get_gene(gene_name=gene_name,datasets=datasets,cells=cell_type)
		<fix/># check if the data is empty:
		<fix/>for dataset in data:
			if data[dataset] == {}:
				return -1</fix></fix>	
		return data

	def new_plot(self,gene_name,cells='ALL'):
		data = self.loader.get_gene(gene_name=gene_name,datasets='ALL',cells=cells)
		for dataset in data:
			if data[dataset] == {}:
				return -1
		return data

	def bar_plot(self,gene_name,cell_name):
		exp_data = self.loader.loadCellSpecific(gene_name,cell_name)
		return exp_data
	"""
	create scatter plot of the expression
	level of the gene in all the cells
	types, and a different plot for each dataset
	"""
	def scatter_plot(self,gene_name):
		data_sets = self.all_sets	
		# check if the gene exist at all.
		if self.loader.findRowMatch(gene_name) == -1:
			return -1
		sets_dict = self.loader.loadGenes([gene_name],data_sets)
		some_data_dict = []
		counter = 0
		for data_set in sets_dict:
			counter +=1
		# add  counnter data sets		
			# create a scatter graph for this set.
			data = sets_dict[data_set]
			head = data['head'][5:]
			gene_data = data[gene_name][5:]
			# splice the data for male and female.

			# create the labels of x axis
			xlabels = []
			for label in head:
				current = label.split('_')[0]
				if current not in xlabels:
					xlabels.append(current)
				else:
					xlabels.append('')

			x_index = []
			for label in xlabels:
				if 0 in x_index:
					if label == '':
						x_index.append(x_index[-1]+self.graph_sub_width_param)
					else:
						x_index.append(x_index[-1]+self.graph_width_param)
				else:
					x_index.append(0)

			female_data = gene_data[::2]
			male_data = gene_data[1::2]
			x_female = []
			x_male = []

			current_indexed = []
			size=0
			# remeber to change the way it arranges data for wierd cells like T4
			# do shenanigans and rearrange some numbers for better apearance			
			for ind, label in enumerate(xlabels):
				if label != '':
					size = len(current_indexed)
					if size != 0:
						for i in current_indexed[:size/2]:
							x_female.append(i)
						for i in current_indexed[size/2:]:
							x_male.append(i)
					current_indexed = []
				current_indexed.append(x_index[ind])
			for i in current_indexed[:size/2]:
				x_female.append(i)
			for i in current_indexed[size/2:]:
				x_male.append(i)

			graph_title = ' '.join([gene_name,'expression level',str(counter)])

			x_labels = xlabels
			data_dic = {}
			data_dic['x_index'] = x_index
			data_dic['x_labels'] = x_labels
			data_dic['title'] = graph_title
			data_dic['x_female'] = x_female
			data_dic['female_data'] = female_data
			data_dic['x_male'] = x_male
			data_dic['male_data'] = male_data
			some_data_dict.append(data_dic)
		return some_data_dict

if __name__ == '__main__':
	# just a test to see if it works
	grapher = Grapher()
	grapher.scatter_plot(['FIRRE' ])
	#grapher.scatter_plot('FIRRE')

	#print grapher.loader.loadGenes(['FIRRE'],['Female_Male_exp_levels_norm.xlsx'])
