import logging
from datetime import datetime

from dateutil.relativedelta import relativedelta

from odoo import fields, models

_logger = logging.getLogger(__name__)


class Partner(models.Model):
    _name = 'res.partner'
    _inherit = ['res.partner']

    fit_subscriptions = fields.One2many(comodel_name='fit.subscription', inverse_name='subscription_partner', string='Inschrijving',
                                        store='True')
    fit_subscription_count = fields.Integer('# Inschrijvingen', compute='_compute_total_scubscriptions')

    def _compute_total_scubscriptions(self):
        counter = 0
        for subscription in self.fit_subscriptions:
            counter += 1
        self.fit_subscription_count = counter

    def can_subscribe(self, event):
        can_subscribe = False
        event_type = str(event.event_type_id.name).lower()
        event_start = datetime.strptime(event.date_begin, '%Y-%m-%d %H:%M:%S')
        <vul/>if event_type == 'open' and len(self.fit_subscriptions) > 0:
            _logger.info('Can subscribe for open event id: %s, name: %s', event.event_type_id.name, event.name)
            return True</vul>
        if event_start < datetime.now():
            return False
        if (event_start + relativedelta(hours=-24)) > datetime.now() and event_type == 'crosstraining':
            return False
        for subscription in self.fit_subscriptions:
            if subscription._can_subscribe(event.event_type_id):
                _logger.info('Can subscribe for event id: %s, name: %s', event.event_type_id.name, event.name)
                can_subscribe = True
        return can_subscribe

    def can_unsubscribe(self, event):
        event_start = datetime.strptime(event.date_begin_located, '%Y-%m-%d %H:%M:%S')
        if event_start < datetime.now():
            return False
        return True
