import urllib

from django.views.generic import TemplateView, FormView, DetailView
from django.urls import reverse

from .entryform import EntryForm, entry_form_config, build_question_flag
from .models import LifeCondition, Benefit, BenefitRequirement


class BenefitOverview(TemplateView):
    template_name = 'core/benefit_overview.html'

    def get_context_data(self):
        data = super().get_context_data()
        data['life_conditions'] = LifeCondition.objects.with_benefits()
        return data


class BenefitClaimView(FormView):
    template_name = 'core/benefit_claim.html'
    form_class = EntryForm

    def get(self, request, *args, **kwargs):
        form = self.get_form()

        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.render_to_response(self.get_context_data())

    def get_form_kwargs(self, *args, **kwargs):
        kwargs = super().get_form_kwargs()
        kwargs['entry_form_config'] = entry_form_config

        question_ids = {str(q['id']) for q in entry_form_config}
        data = {
            f'{item}': f'{value}' for item, value in self.request.GET.items() if item in question_ids
        }

        if data:
            kwargs['data'] = data

        return kwargs

    def form_valid(self, form):
        selected_flags = []

        # Assemble query
        for question in entry_form_config:
            flag = form.cleaned_data.get(str(question['id']), False)

            if flag:
                selected_flags.append(getattr(BenefitRequirement.flags, build_question_flag(question)))

        return self.render_to_response({
            'form': form,
            'submitted': True,
            'claimable_benefits': Benefit.objects.find_claimable(selected_flags),
        })


class BenefitDetailView(DetailView):
    model = Benefit
    template_name = 'core/benefit_detail.html'

    def get_context_data(self, *args, **kwargs):
        data = super().get_context_data(*args, **kwargs)

        <fix/>back = self.request.GET.get('back', None)
        parsed_back_url = urllib.parse.urlparse(back)

        # We only allow blank scheme, e.g. relative urls to avoid reflected XSS
        if back is not None and parsed_back_url.scheme == "":
            data['back_link'] = back</fix>

        return data
