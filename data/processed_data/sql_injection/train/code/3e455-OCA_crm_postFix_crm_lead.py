# -*- coding: utf-8 -*-
# Copyright (C) 2004-today OpenERP SA (<http://www.openerp.com>)
# Copyright (C) 2017 Tecnativa - Vicent Cubells
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl).

from openerp import api, fields, models


class CrmLead(models.Model):
    _inherit = "crm.lead"

    phonecall_ids = fields.One2many(
        comodel_name='crm.phonecall',
        inverse_name='opportunity_id',
        string='Phonecalls',
    )
    phonecall_count = fields.Integer(
        <fix/>compute='_compute_phonecall_count',</fix>
        string="Phonecalls",
    )

    @api.multi
    <fix/>def _compute_phonecall_count(self):</fix>
        for lead in self:
            lead.phonecall_count = self.env[
                'crm.phonecall'].search_count(
                [('opportunity_id', '=', lead.id)])
