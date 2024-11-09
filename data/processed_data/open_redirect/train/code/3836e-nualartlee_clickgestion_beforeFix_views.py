<vul/></vul>from django.apps import apps
from clickgestion.transactions.forms import TransactionEditForm, TransactionPayForm
from clickgestion.transactions.models import BaseConcept, Transaction
from django.shortcuts import get_object_or_404, render, redirect, reverse
from django.utils.translation import gettext, gettext_lazy
from clickgestion.transactions.filters import ConceptFilter, TransactionFilter
from clickgestion.core.utilities import invalid_permission_redirect
from django.views.generic import ListView
from django.contrib.auth.decorators import login_required
from pure_pagination.mixins import PaginationMixin
from django.http import HttpResponse, QueryDict
from django.conf import settings
from django.utils import timezone
from django_xhtml2pdf.utils import generate_pdf


@login_required()
def concept_delete(request, *args, **kwargs):
    extra_context = {}

    # Check permissions

    # Get the concept and form
    concept, concept_form = get_concept_and_form_from_kwargs(**kwargs)
    extra_context['concept'] = concept

    # Get the transaction
    transaction = concept.transaction
    if transaction.closed:
        return redirect('message', message=gettext('Transaction Closed'))
    extra_context['transaction'] = transaction

    # Use default delete view
    extra_context['header'] = gettext('Delete {}?'.format(concept.concept_type))
    extra_context['message'] = concept.description_short
    extra_context['next'] = request.META['HTTP_REFERER']

    # POST
    if request.method == 'POST':
        default_next = reverse('transaction_detail', kwargs={'transaction_code': concept.transaction.code})
        concept.delete()
        next_page = request.POST.get('next', default_next)
        return redirect(next_page)

    # GET
    else:
        return render(request, 'core/delete.html', extra_context)


@login_required()
def concept_detail(request, *args, **kwargs):
    extra_context = {}

    # Check permissions

    # Get the concept and form
    concept, concept_form = get_concept_and_form_from_kwargs(**kwargs)
    extra_context['concept'] = concept

    # Get the transaction
    transaction = concept.transaction
    extra_context['transaction'] = transaction

    return render(request, 'transactions/concept_detail.html', extra_context)


@login_required()
def concept_edit(request, *args, **kwargs):
    extra_context = {}

    # Check permissions

    # Get the concept and form
    concept, concept_form = get_concept_and_form_from_kwargs(**kwargs)
    extra_context['concept'] = concept

    # Get the transaction
    transaction = concept.transaction
    if transaction.closed:
        return redirect('message', message=gettext('Transaction Closed'))
    extra_context['transaction'] = transaction

    # POST
    if request.method == 'POST':
        form = concept_form(request.POST, instance=concept)
        if form.is_valid():
            form.save()
            return redirect('transaction_edit', transaction_code=transaction.code)

        else:
            extra_context['form'] = form
            return render(request, 'transactions/concept_edit.html', extra_context)

    # GET
    else:

        # Get the form
        form = concept_form(instance=concept)
        extra_context['form'] = form
        return render(request, 'transactions/concept_edit.html', extra_context)


class ConceptList(PaginationMixin, ListView):

    template_name = 'transactions/concept_list.html'
    model = BaseConcept
    context_object_name = 'concepts'
    paginate_by = 8
    # ListView.as_view will pass custom arguments here
    queryset = None
    header = gettext_lazy('Concepts')
    request = None
    filter = None
    filter_data = None
    is_filtered = False

    def get(self, request, *args, **kwargs):
        # First

        # Check permissions
        if not request.user.is_authenticated:
            return invalid_permission_redirect(request)

        # Get arguments
        self.request = request
        self.filter_data = kwargs.pop('filter_data', {})

        # Call super
        return super().get(self, request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        # Third

        # Call the base implementation first
        context = super().get_context_data(**kwargs)

        # Add data
        context['header'] = self.header
        context['filter'] = self.filter
        context['is_filtered'] = self.is_filtered

        return context

    def get_queryset(self):
        # Second

        # Create filter querydict
        data = QueryDict('', mutable=True)
        # Add filters passed from view
        data.update(self.filter_data)
        # Add filters selected by user
        data.update(self.request.GET)

        # Record as filtered
        self.is_filtered = False
        if len([k for k in data.keys() if k != 'page']) > 0:
            self.is_filtered = True

        # Add filters by permission

        # Filter the queryset
        self.filter = ConceptFilter(data)
        self.queryset = self.filter.qs.select_related('transaction') \
            .prefetch_related('value__currency') \
            .order_by('-id') # 79q 27ms

        # Return
        return self.queryset


def get_available_concepts(employee, transaction):
    """
    Get a list of the available concepts that can be added to the given transaction.

    :param employee: The employee executing the transaction (current user)
    :param transaction: The open transaction
    :return: A list of dictionaries.
    """

    # get permissions according to transaction
    concepts_permitted_by_transaction = transaction.get_all_permissions()

    # get permissions according to employee
    concepts_permitted_by_employee = employee.get_all_permissions()

    # create the list of permitted concepts
    available_concepts = []
    for concept in settings.CONCEPTS:
        permission = concept.replace('.','.add_')
        concept_model = apps.get_model(concept)

        # Skip this concept if not permitted by the user
        if not permission in concepts_permitted_by_employee:
            continue

        # set default values
        disabled = False
        url = concept_model._url.format('new/{}'.format(transaction.code))

        # disable this concept if not permitted by the transaction
        if not permission in concepts_permitted_by_transaction:
            disabled = True
            url = '#'

        # add to the list
        available_concepts.append(
            {
                'name': concept_model._meta.verbose_name,
                'url': url,
                'disabled': disabled,
            }
        )

    return available_concepts


def get_transaction_from_kwargs(**kwargs):
    # Get the transaction
    transaction_code = kwargs.get('transaction_code', None)
    transaction = get_object_or_404(Transaction, code=transaction_code)
    return transaction


def get_concept_and_form_from_kwargs(**kwargs):

    # Get the form
    concept_form = kwargs.get('concept_form', None)

    # Get the concept class
    concept_class = concept_form._meta.model

    # If a transaction code is provided, this is a new concept
    transaction_code = kwargs.get('transaction_code', None)
    if transaction_code:
        transaction = get_transaction_from_kwargs(**kwargs)
        return concept_class(transaction=transaction), concept_form

    # Get the existing concept
    concept_code = kwargs.get('concept_code', None)
    concept = get_object_or_404(concept_class, code=concept_code)
    return concept, concept_form


def transaction_delete(request, *args, **kwargs):
    extra_context = {}

    # Check permissions
    if not request.user.is_authenticated:
        return invalid_permission_redirect(request)

    # Get the object
    transaction_code = kwargs.get('transaction_code', None)
    transaction = get_object_or_404(Transaction, code=transaction_code)
    extra_context['transaction'] = transaction

    # Use default delete view
    extra_context['header'] = gettext('Delete Transaction?')
    extra_context['message'] = transaction.description_short
    extra_context['next'] = request.META['HTTP_REFERER']

    # POST
    if request.method == 'POST':
        default_next = reverse('transactions_open')
        transaction.delete()
        next_page = request.POST.get('next', default_next)
        return redirect(next_page)

    # GET
    else:
        return render(request, 'core/delete.html', extra_context)


@login_required
def transaction_detail(request, *args, **kwargs):
    extra_context = {}

    # Check permissions
    if not request.user.is_authenticated:
        return invalid_permission_redirect(request)

    # Get the transaction
    transaction_code = kwargs.get('transaction_code', None)
    transaction = get_object_or_404(Transaction, code=transaction_code)
    extra_context['transaction'] = transaction
    return render(request, 'transactions/transaction_detail.html', extra_context)


@login_required
def transaction_edit(request, *args, **kwargs):
    extra_context = {}

    # Check permissions
    if not request.user.is_authenticated:
        return invalid_permission_redirect(request)

    # Get the transaction
    transaction_code = kwargs.get('transaction_code', None)
    if not transaction_code:
        transaction = Transaction.objects.create(employee=request.user)
        return redirect('transaction_edit', transaction_code=transaction.code)

    transaction = get_object_or_404(Transaction, code=transaction_code)
    extra_context['transaction'] = transaction

    # Check that the transaction is open
    if transaction.closed:
        return redirect('message', message=gettext('Transaction Closed'))

    # Get available concepts to add
    available_concepts = get_available_concepts(request.user, transaction)
    extra_context['available_concepts'] = available_concepts

    # POST
    if request.method == 'POST':
        form = TransactionEditForm(request.POST, instance=transaction)
        valid = form.is_valid()

        # Delete and go home
        # Note that the form.data value is still a string before validating
        if form.data['cancel_button'] == 'True':
            transaction.delete()
            return redirect('index')

        # If valid
        if valid:

            # Save the transaction
            transaction.save()

            # Finish later
            if form.cleaned_data['save_button']:
                return redirect('transaction_detail', transaction_code=transaction.code)

            # Proceed to pay
            return redirect('transaction_pay', transaction_code=transaction.code)

        else:
            extra_context['form'] = form
            return render(request, 'transactions/transaction_edit.html', extra_context)

    # GET
    else:

        # Create the form
        form = TransactionEditForm(instance=transaction)
        extra_context['form'] = form
        return render(request, 'transactions/transaction_edit.html', extra_context)


class TransactionList(PaginationMixin, ListView):

    model = Transaction
    context_object_name = 'transactions'
    paginate_by = 8
    # ListView.as_view will pass custom arguments here
    queryset = None
    header = gettext_lazy('Transactions')
    request = None
    filter = None
    filter_data = None
    is_filtered = False

    def get(self, request, *args, **kwargs):
        # First

        # Check permissions
        if not request.user.is_authenticated:
            return invalid_permission_redirect(request)

        # Get arguments
        self.request = request
        self.filter_data = kwargs.pop('filter_data', {})

        # Call super
        return super().get(self, request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        # Third

        # Call the base implementation first
        context = super().get_context_data(**kwargs)

        # Add data
        context['header'] = self.header
        context['filter'] = self.filter
        context['is_filtered'] = self.is_filtered

        return context

    def get_queryset(self):
        # Second

        # Create filter querydict
        data = QueryDict('', mutable=True)
        # Add filters passed from view
        data.update(self.filter_data)
        # Add filters selected by user
        data.update(self.request.GET)

        # Record as filtered
        self.is_filtered = False
        if len([k for k in data.keys() if k != 'page']) > 0:
            self.is_filtered = True

        # Add filters by permission

        # Filter the queryset
        self.filter = TransactionFilter(data)
        self.queryset = self.filter.qs.select_related('cashclose')\
            .prefetch_related('concepts__value__currency') \
            .order_by('-id')  # 79q 27ms

        # Return
        return self.queryset

    def post(self, request, *args, **kwargs):

        # Check permissions
        if not request.user.is_authenticated:
            return invalid_permission_redirect(request)

        print_transaction = request.POST.get('print_transaction', None)
        if print_transaction:
            # Get the transaction
            transaction = get_object_or_404(Transaction, code=print_transaction)
            # Create an http response
            resp = HttpResponse(content_type='application/pdf')
            resp['Content-Disposition'] = 'attachment; filename="{}.pdf"'.format(transaction.code)
            # Set context
            context = {
                'transaction': transaction,
            }
            # Generate the pdf
            result = generate_pdf('transactions/invoice.html', file_object=resp, context=context)
            return result

        # Return same
        request.method = 'GET'
        return self.get(request, *args, **kwargs)



@login_required
def transaction_pay(request, *args, **kwargs):
    extra_context = {}

    # Check permissions
    if not request.user.is_authenticated:
        return invalid_permission_redirect(request)

    # Get the transaction
    transaction_code = kwargs.get('transaction_code', None)
    transaction = get_object_or_404(Transaction, code=transaction_code)
    extra_context['transaction'] = transaction

    # Check that the transaction is open
    if transaction.closed:
        return redirect('message', message=gettext('Transaction Closed'))

    # Get required payment fields

    # POST
    if request.method == 'POST':
        form = TransactionPayForm(request.POST, instance=transaction)
        valid = form.is_valid()

        # If cancel has been set, delete and go home
        # Note that value is still string before validating
        if form.data['cancel_button'] == 'True':
            transaction.delete()
            return redirect('index')

        # If valid
        if valid:

            # Close the transaction
            if form.cleaned_data['confirm_button']:
                transaction.closed = True
                transaction.closed_date = timezone.datetime.now()
                transaction.save()
                return redirect('transaction_detail', transaction_code=transaction.code)

            # Save the transaction
            if form.cleaned_data['save_button']:
                transaction.save()
                return redirect('transaction_detail', transaction_code=transaction.code)

        else:
            extra_context['form'] = form
            return render(request, 'transactions/transaction_pay.html', extra_context)

    # GET
    else:

        # Create the form
        form = TransactionPayForm(instance=transaction)
        extra_context['form'] = form
        return render(request, 'transactions/transaction_pay.html', extra_context)


def transactions_open(request, *args, **kwargs):

    # Check permissions
    if not request.user.is_authenticated:
        return invalid_permission_redirect(request)

    # Set initial filter data
    filter_data = {
        'closed': False,
    }

    # Return
    <vul/>listview = TransactionList.as_view()
    return listview(request, filter_data=filter_data)</vul>
