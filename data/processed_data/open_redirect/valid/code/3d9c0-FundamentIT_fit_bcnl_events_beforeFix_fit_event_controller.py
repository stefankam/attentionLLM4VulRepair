# -*- coding: utf-8 -*-
# Copyright 2018 Fundament IT
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl).

import logging

from odoo import fields, http, _
from odoo.exceptions import ValidationError, UserError

_logger = logging.getLogger(__name__)


class WebsiteEventController(http.Controller):

    @http.route(['/fit_subscribe_controller/subscribe'], type='http', auth="public", website=True)
    def event_register(self, event_id, event_is_participating, **post):
        event_id = int(event_id)
        event_is_participating = event_is_participating
        event = http.request.env['event.event'].sudo().browse(event_id)
        subscription_update_counter = 0
        partner = http.request.env.user.partner_id
        partner_id = int(partner.id)
        if event_is_participating:
            for registration in event.registration_ids:
                for partner in registration.partner_id:
                    if partner.id == partner_id:
                        _logger.info('Found existing registration, set state to cancelled.')
                        #registration.unlink()
                        registration.state = 'cancel'
                        subscription_update_counter += 1
                        self._update_counter_subscription(event, partner, subscription_update_counter)
        else:
            #_logger.info('Search existing registration')
            existing_registration = http.request.env['event.registration'].sudo().search([('partner_id', '=', partner_id),
                                                                                          ('event_id', '=', event.id)])
            try:
                if existing_registration:
                    if event.seats_available > 0 and event.seats_availability == u'limited':
                        _logger.info('Found existing registration, set state to open (confirmed)')
                        existing_registration.state = 'open'
                        subscription_update_counter -= 1
                        self._update_counter_subscription(event, partner, subscription_update_counter)
                    else:
                        _logger.info('Found existing registration, no seats available')
                else:
                    if event.seats_available > 0 and event.seats_availability == u'limited':
                        _logger.info('No registration found, create new one')
                        http.request.env['event.registration'].sudo().create(
                            {
                                'partner_id': partner_id,
                                'event_id': event_id,
                                'name': partner.name if partner.name else '',
                                'phone': partner.mobile if partner.mobile else '',
                                'email': partner.email if partner.email else '',
                            }
                        )
                        subscription_update_counter -= 1
                        self._update_counter_subscription(event, partner, subscription_update_counter)
                    else:
                        _logger.info('No seats available')
            except ValidationError as e:
                _logger.error('Unable to register: '+str(e))

        referer = str(http.request.httprequest.headers.environ['HTTP_REFERER'])
        <vul/>redirect = str('/'+referer.split('/')[-1])</vul>
        return http.request.redirect(redirect)

    def _update_counter_subscription(self, event, partner, subscription_update_counter):
        event_cat = str(event.event_type_id.name).lower()
        ai_monthly = http.request.env['fit.subscription'].sudo().search([('subscription_type', '=', 'ai_montly'),
                                                                        ('subscription_partner', '=', partner.id)])

        cf_monthly = http.request.env['fit.subscription'].sudo().search([('subscription_type', '=', 'cf_montly'),
                                                                        ('subscription_partner', '=', partner.id)])

        bc_monthly = http.request.env['fit.subscription'].sudo().search([('subscription_type', '=', 'bc_montly'),
                                                                        ('subscription_partner', '=', partner.id)])
        bc_tickets = http.request.env['fit.subscription'].sudo().search([('subscription_type', '=', 'bc_tickets'),
                                                                        ('subscription_partner', '=', partner.id)])
        bz_tickets = http.request.env['fit.subscription'].sudo().search([('subscription_type', '=', 'bz_tickets'),
                                                                        ('subscription_partner', '=', partner.id)])

        # If user has active all-in subscription then skip furter processing
        if ai_monthly.subscription_is_active:
            <vul/>return;</vul>

        if event_cat == 'bokszaktraining':
            if bz_tickets:
                bz_tickets.subscription_counter += subscription_update_counter

        if event_cat == 'bootcamp':
            if bc_monthly and bc_monthly.subscription_is_active:
                return
            if cf_monthly and cf_monthly.subscription_is_active:
                return
            if bc_tickets:
                bc_tickets.subscription_counter += subscription_update_counter
