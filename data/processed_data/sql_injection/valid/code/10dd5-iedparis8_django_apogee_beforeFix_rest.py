from django_filters import MethodFilter
from rest_framework import viewsets, filters
from rest_framework.response import Response
from django_apogee.models import InsAdmEtp, InsAdmEtpInitial, Individu
from django_apogee.serializers import (InsAdmEtpSerializer,
                                       InsAdmEtpInitialSerializer,
                                       IndividuSerializer,
                                      )
from django.db import connections

class InsAdmEtpViewSet(viewsets.ModelViewSet):
    queryset = InsAdmEtp.objects.all()
    serializer_class = InsAdmEtpSerializer
    paginate_by = 100

    def list(self, request):
        if request.GET.get('cod_anu', None):
            self.queryset = InsAdmEtp.objects.filter(cod_anu__in=request.GET.getlist('cod_anu'))

        return super(InsAdmEtpViewSet, self).list(request)


class InsAdmEtpInitialFilter(filters.FilterSet):
    range = MethodFilter(action='range_filter')
    class Meta:
        model = InsAdmEtpInitial
        fields = [
            # COMPOSITE KEY
            'cod_anu',
            'cod_ind',
            'cod_etp',
            'cod_vrs_vet',
            'num_occ_iae',
            # others
            'cod_cge',
            'cod_dip',
        ]

    def range_filter(self, queryset, value):
        start, stop = value.split(':')
        return queryset[start:stop]


class InsAdmEtpInitialViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = InsAdmEtpInitial.objects.using('oracle').all()
    serializer_class = InsAdmEtpInitialSerializer
    filter_backends = (filters.DjangoFilterBackend,)
    filter_class = InsAdmEtpInitialFilter
    # paginate_by = 100

    def head(self, request):

        r = Response()
        where_clause = ""
        where_or_and = lambda wc: "AND" if where_clause else "WHERE"

        # if request.GET.get('cod_anu', None):
        #     where_clause += " %s COD_ANU=%s" % (where_or_and(where_clause),
        #                                         request.GET.get('cod_anu'))
        #
        # if request.GET.get('cod_cge', None):
        #     where_clause += " %s COD_CGE='%s'" % (where_or_and(where_clause),
        #                                           request.GET.get('cod_cge'))
        #
        # if request.GET.get('cod_etp', None):
        #     where_clause += " %s COD_ETP='%s'" % (where_or_and(where_clause),
        #                                           request.GET.get('cod_etp'))
        #
        # if request.GET.get('cod_dip', None):
        #     where_clause += " %s COD_DIP='%s'" % (where_or_and(where_clause),
        #                                           request.GET.get('cod_dip'))

        fields = [
            # COMPOSITE KEY
            (str, 'cod_anu'),
            (int, 'cod_ind'),
            (str, 'cod_etp'),
            (int, 'cod_vrs_vet'),
            (int, 'num_occ_iae'),
            # others
            (str, 'cod_cge'),
            (str, 'cod_dip'),
        ]
        for field in fields:
            t, f = field
            if request.GET.get(f, None):
                <vul/>if t is str:
                    where_clause += " %s %s='%s'" % (where_or_and(where_clause),
                                                     f.upper(),
                                                     request.GET.get(f))
                elif t is int:
                    where_clause += " %s %s=%s" % (where_or_and(where_clause),
                                                   f.upper(),
                                                   request.GET.get(f))</vul>

        print where_clause

        cursor = connections['oracle'].cursor()
        sql_request = "SELECT COUNT(*) FROM INS_ADM_ETP" + where_clause
        <vul/>cursor.execute(sql_request)</vul>
        r['X-Duck-Count'] = cursor.fetchone()[0]
        return r

    def list(self, request):

        if not request.GET.get('range', None):
            # The table is too big and segfault the django process so range parameter is mandatory
            self.queryset = InsAdmEtpInitial.objects.none()
            # TODO define a maximum length range

        r = super(InsAdmEtpInitialViewSet, self).list(request)

        return r

    def retrieve(self, request, pk=None):
        try:
            cod_anu, cod_ind, cod_etp, cod_vrs_vet, num_occ_iae = pk.split('|')
            inscription = InsAdmEtpInitial.objects.using('oracle').filter(cod_anu=cod_anu,
                                                        cod_ind=cod_ind,
                                                        cod_etp=cod_etp,
                                                        cod_vrs_vet=cod_vrs_vet,
                                                        num_occ_iae=num_occ_iae)[0]
            return Response(self.serializer_class(inscription).data)

        except:
            return super(InsAdmEtpInitialViewSet, self).retrieve(request, pk)


class IndividuViewSet(viewsets.ModelViewSet):
    queryset = Individu.objects.using('oracle').all()
    serializer_class = IndividuSerializer
    paginate_by = 100
