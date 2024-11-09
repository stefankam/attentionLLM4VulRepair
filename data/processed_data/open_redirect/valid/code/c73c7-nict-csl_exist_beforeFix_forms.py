from django import forms

class SearchForm(forms.Form):
    <vul/>keyword = forms.CharField(label='', max_length=100, required=True)</vul>
    keyword.widget.attrs['class'] = 'form-control mr-sm-2 my-2'
    keyword.widget.attrs['placeholder'] = 'Lookup URL'
