from django.shortcuts import render, get_object_or_404, reverse, redirect, HttpResponse
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import CreateView, ListView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from .forms import ShortUrlForm, JustURLForm, CategoryModelForm, ManyURLSForm, JustULRUpdateForm, \
    CategoryUpdateModelForm, CounterCountingForm
from .models import JustURL, Category, ClickTracking
from .utils import create_short_url, token_generator, generate_csv, get_client_ip, check_input_url
from Shortener_Project.settings import SHORTCODE
import re


class HomeView(View):
    def get(self, request, *args, **kwargs):
        form = ShortUrlForm()
        return render(request, 'home.html', {'form': form})

    def post(self, request, *args, **kwargs):
        form = ShortUrlForm(request.POST or None)
        if form.is_valid():
            url = form.cleaned_data['input_url']
            category = form.cleaned_data['category']
            created = JustURL.objects.create(input_url=url, category=category)
            short_url = create_short_url(created)
            created.short_url = f'{request.get_host()}/{short_url}'
            created.save()
            if request.user.is_superuser:
                return redirect(reverse('url-detail-view', kwargs={'pk': created.pk}))
            return redirect(reverse('success-url-view', kwargs={'pk': created.pk}))
        return render(request, 'home.html', {'form': form})


class SuccessUrlView(View):
    def get(self, request, pk, *args, **kwargs):
        object = JustURL.objects.get(pk=pk)
        form = CounterCountingForm()
        return render(request, 'success-url-view.html', {'object': object,
                                                         'form': form})

    def post(self, request, pk, *args, **kwargs):
        object = JustURL.objects.get(pk=pk)
        form = CounterCountingForm(request.POST or None)
        if form.is_valid():
            ip = get_client_ip(request)
            client_agent = request.META['HTTP_USER_AGENT']
            clicktracker = ClickTracking.objects.create(
                client_ip=ip,
                user_agent=client_agent,
            )
            clicktracker.url.add(object)
            clicktracker.save()
            <fix/><fix/>token = object.short_url[-SHORTCODE:]
            return link_redirect(request, token)</fix></fix>
        return redirect('home-view')


class URLDetailView(LoginRequiredMixin, View):
    def get(self, request, pk, *args, **kwargs):
        form = CounterCountingForm()
        object = JustURL.objects.get(pk=pk)
        return render(request, 'url-detail-view.html', {'object': object,
                                                        'form': form})

    def post(self, request, pk, *args, **kwargs):
        object = JustURL.objects.get(pk=pk)
        form = CounterCountingForm(request.POST or None)
        if form.is_valid():
            ip = get_client_ip(request)
            client_agent = request.META['HTTP_USER_AGENT']
            clicktracker = ClickTracking.objects.create(
                client_ip=ip,
                user_agent=client_agent,
            )
            clicktracker.url.add(object)
            clicktracker.save()
            token = object.short_url[-SHORTCODE:]
            return link_redirect(request, token)
        return render(request, 'url-detail-view.html', {'object': object,
                                                        'form': form})


class URLUpdateView(LoginRequiredMixin, UpdateView):
    queryset = JustURL.objects.all()
    form_class = JustULRUpdateForm
    template_name = 'url-update-view.html'


class URLDeleteView(LoginRequiredMixin, DeleteView):
    model = JustURL
    template_name = 'url-delete-view.html'
    success_url = reverse_lazy('home-view')


class CustomShortURLCreateView(View):
    def get(self, request, *args, **kwargs):
        form = JustURLForm()
        return render(request, 'custom-short-url.html', {'form': form})

    def post(self, request, *args, **kwargs):
        form = JustURLForm(request.POST or None)
        if form.is_valid():
            url = form.cleaned_data['input_url']
            short_url = form.cleaned_data['short_url']
            category = form.cleaned_data['category']
            if JustURL.objects.filter(short_url__contains=short_url).exists():
                message = 'Token is already in use'
                return render(request, 'custom-short-url.html', {'form': JustURLForm,
                                                                 'message': message})
            created = JustURL.objects.create(input_url=url, short_url=f'{request.get_host()}/{short_url}',
                                             category=category)
            created.save()
            if request.user.is_superuser:
                return redirect(reverse('url-detail-view', kwargs={'pk': created.pk}))
            return redirect(reverse('success-url-view', kwargs={'pk': created.pk}))
        return render(request, 'home.html', {'form': form})


class ShortManyURLSView(View):
    def get(self, request, *args, **kwargs):
        form = ManyURLSForm()
        return render(request, 'short-many-urls.html', {'form': form})

    def post(self, request, *args, **kwargs):
        form = ManyURLSForm(request.POST or None)
        if form.is_valid():
            urls = form.cleaned_data['input_url']
            urls_list = re.findall(r"[\w.']+", urls)
            data_list = []
            for url in urls_list:
                result = check_input_url(url)
                instance = JustURL.objects.create(input_url=result,
                                                  short_url=f'{request.get_host()}/{token_generator()}')
                instance.save()
                data = [instance.input_url, instance.short_url]
                data_list.append(data)

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="many_urls.csv"'
            return generate_csv(data_list, response)
        return redirect('home-view')


class CategoryCreateView(LoginRequiredMixin, CreateView):
    template_name = 'category-create-view.html'
    form_class = CategoryModelForm


class CategoryListView(LoginRequiredMixin, ListView):
    queryset = Category.objects.all().order_by('name')
    template_name = 'category-list-view.html'
    paginate_by = 15

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        quantity = 0
        urls_without_category = JustURL.objects.filter(category=None).count()
        print(urls_without_category)
        queryset = Category.objects.all()
        for cat in queryset:
            quantity += cat.justurl_set.all().count()
        context['number_of_links'] = quantity
        context['urls_without_category'] = urls_without_category
        return context


class CategoryDetailView(LoginRequiredMixin, View):
    def get(self, request, pk, *args, **kwargs):
        object = Category.objects.get(pk=pk)
        visits = sum(link.count for link in object.justurl_set.all())
        return render(request, 'category-detail-view.html', {'object': object,
                                                             'visits': visits})


class CategoryUpdateView(LoginRequiredMixin, UpdateView):
    queryset = Category.objects.all()
    form_class = CategoryUpdateModelForm
    template_name = 'category-update-view.html'
    success_url = reverse_lazy('category-list-view')


class CategoryDeleteView(LoginRequiredMixin, DeleteView):
    model = Category
    template_name = 'category-delete-view.html'
    success_url = reverse_lazy('category-list-view')


class ClickTrackingDetailView(LoginRequiredMixin, View):
    def get(self, request, pk, *args, **kwargs):
        object = get_object_or_404(JustURL, pk=pk)
        reports = object.clicktracking_set.all().order_by('timestamp')
        return render(request, 'clicktracking-detail-view.html', {'object': object,
                                                                  'reports': reports})


<fix/>def link_redirect(request, token):
    instance = JustURL.objects.get(short_url__icontains=token)
    instance.count += 1
    instance.save()</fix>
    return redirect(instance.input_url)
