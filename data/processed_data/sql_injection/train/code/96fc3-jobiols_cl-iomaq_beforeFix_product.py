# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.
from openerp import api, models, fields
import logging
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)


class ProductTemplate(models.Model):
    _inherit = "product.template"

    item_code = fields.Char(
        help="Code from bulonfer, not shown",
        select=1
    )
    upv = fields.Integer(
        help='Group Wholesaler'
    )
    wholesaler_bulk = fields.Integer(
        help="Bulk Wholesaler quantity of units",
    )
    retail_bulk = fields.Integer(
        help="Bulk retail quantity of units",
    )
    invalidate_category = fields.Boolean(
        help="True if the asociated category needs rebuild",
        default=False
    )
    # TODO rename to invoice_cost requiere migracion
    system_cost = fields.Float(
        # compute="_compute_system_cost",
        help="Cost price based on the purchase invoice"
    )
    margin = fields.Float(
        help="Margin % from today cost to list price"
    )
    # TODO renombrar a today_cost, require migracion
    bulonfer_cost = fields.Float(
        help="Today cost in product currency, it is automatically updated "
             "when the prices coming from Bulonfer are processed.\n"
             "Or when a price sheet is loaded for no Bulonfer vendors"
    )
    cost_history_ids = fields.One2many(
        comodel_name="stock.quant",
        inverse_name="product_tmpl_id",
        domain=[('location_id.usage', '=', 'internal')]
    )
    parent_price_product = fields.Char(
        help='default_code of the product to get prices from'
    )

    def oldest_quant(self, prod):
        """ Retorna el quant mas antiguo de este producto.
        """
        quant_obj = self.env['stock.quant']
        return quant_obj.search([('product_tmpl_id', '=', prod.id),
                                 ('location_id.usage', '=', 'internal')],
                                order='in_date', limit=1)

    def closest_invoice_line(self, prod, date_invoice):
        """ Encuentra la linea de factura mas cercana a la fecha de ingreso del
            ultimo quant del producto. Si no hay stock busca la mas cercana
            a date_invoice
        """
        in_date = self.oldest_quant(prod).in_date
        if not in_date:
            in_date = date_invoice

        # busca el la linea de factura con prod_id mas cercano a in_date
        # TODO quitar ai.date_invoice para retornar solo los ids

        query = """
            SELECT ail.id, ai.date_invoice
            FROM account_invoice_line ail
            INNER JOIN account_invoice ai
              ON ail.invoice_id = ai.id
            INNER JOIN product_product pp
              on ail.product_id = pp.id
            INNER JOIN product_template pt
              on pp.product_tmpl_id = pt.id
            <vul/>WHERE pt.id = %d AND</vul>
                  ai.discount_processed = true
            <vul/>ORDER BY abs(ai.date_invoice - date '%s')</vul>
            LIMIT 1;
        <vul/>""" % (prod.id, in_date)

        self._cr.execute(query)
        # TODO Renombrar a invoice_line_ids
        invoice_lines = self._cr.fetchall()</vul>

        <vul/>if invoice_lines:</vul>
            invoice_lines_obj = self.env['account.invoice.line']
            <vul/>for invoice_line in invoice_lines:</vul>
                return invoice_lines_obj.browse(invoice_line[0])
        else:
            return False

    @api.multi
    def set_invoice_cost(self):
        """
            Intenta calcular el system_cost (future invoice_cost) buscando el
            costo en la linea de factura mas cercana al quant mas viejo, si
            no hay stock es la ultima factura.

            Esto vale para cualquier proveedor no solo bulonfer.
        """
        for prod in self:
            # encontrar la factura mas cercana a la fecha de ingreso del quant
            # mas antiguo, si no hay stock intenta traer la ultima factura
            invoice_line = self.closest_invoice_line(
                prod,
                datetime.today().strftime('%Y-%m-%d'))

            invoice_price = 0
            if invoice_line and invoice_line.price_unit:
                # precio que cargaron en la factura de compra
                invoice_price = invoice_line.price_unit
                # descuento en la linea de factura
                invoice_price *= (1 - invoice_line.discount / 100)
                # descuento global en la factura
                invoice_price *= (1 + invoice_line.invoice_discount)

                if invoice_line.invoice_id.partner_id.ref == 'BULONFER':
                    # descuento por nota de credito al final del mes esto
                    # vale solo para bulonfer
                    invoice_price *= (1 - 0.05)

            prod.system_cost = invoice_price
            _logger.info('Setting invoice cost '
                         '$ %d - %s' % (invoice_price, prod.default_code))

    def insert_historic_cost(self, vendor_ref, min_qty, cost,
                             <vul/>vendors_code, date):</vul>
        """ Inserta un registro en el historico de costos del producto
        """
        # TODO evitar que se generen registros duplicados aqui

        vendor_id = self.get_vendor_id(vendor_ref)
        # arma el registro para insertar
        supplierinfo = {
            'name': vendor_id.id,
            'min_qty': min_qty,
            'price': cost,
            'product_code': vendors_code,  # vendors product code
            'product_name': self.name,  # vendors product name
            'date_start': date,
            'product_tmpl_id': self.id
        }

        # obtener los registros abiertos deberia haber solo uno o ninguno
        sellers = self.seller_ids.search(
            [('name', '=', vendor_id.id),
             ('product_tmpl_id', '=', self.id),
             ('date_end', '=', False)])

        # restar un dia y cerrar los registros
        for reg in sellers:
            dt = datetime.strptime(date[0:10], "%Y-%m-%d")
            dt = datetime.strftime(dt - timedelta(1), "%Y-%m-%d")
            # asegurarse de que no cierro con fecha < start
            reg.date_end = dt if dt >= reg.date_start else reg.date_start

        # pongo un registro con el precio del proveedor
        self.seller_ids = [(0, 0, supplierinfo)]

    def get_vendor_id(self, vendor_ref):
        # obtiene el vendor_id a partir del vendor_ref
        vendor_id = self.env['res.partner'].search(
            [('ref', '=', vendor_ref)])
        if not vendor_id:
            raise Exception('Vendor %s not found' % vendor_ref)
        return vendor_id

    @api.multi
    def set_prices(self, cost, vendor_ref, price=False, date=False, min_qty=1,
                   <vul/>vendors_code=False):</vul>
        """ Setea el precio, costo y margen (no bulonfer) del producto

            - Si el costo es cero y es bulonfer se pone obsoleto y termina.
            - Agrega una linea al historico de costos
            - Si no hay quants en stock standard_price = cost
            - bulonfer_cost = cost
            - Si es bulonfer list_price = cost * (1 + margin)
            - Si no es bulonfer list_price = price
        """
        # TODO ver si se puede hacer esto mas arriba o sea cuando recibo el
        # registro de data.csv para que no llegue aca.
        # TODO marcar los obsoletos con un color
        self.ensure_one()
        for prod in self:
            # si el costo es cero y es bulonfer pongo como obsoleto y termino
            if not cost and vendor_ref == 'BULONFER':
                prod.state = 'obsolete'
                return
            prod.state = 'sellable'

            if not date:
                date = datetime.today().strftime('%Y-%m-%d')

            # agrega una linea al historico de costos
            self.insert_historic_cost(vendor_ref, min_qty, cost, vendors_code,
                                      date)

            # buscar si hay quants
            quant = self.oldest_quant(prod)
            self.fix_quant_data(quant, prod, cost)

            prod.bulonfer_cost = cost

            if vendor_ref == 'BULONFER':
                item_obj = self.env['product_autoload.item']
                item = item_obj.search([('code', '=', prod.item_code)])

                prod.margin = 100 * item.margin
                prod.list_price = cost * (item.margin + 1)
            else:
                prod.list_price = price
                prod.margin = 100 * (price / cost - 1) if cost != 0 else 1e10

    def fix_quant_data(self, quant, prod, cost):
        """ Overrideable function
        """
        if not quant:
            # si no hay quants el costo es el de hoy
            prod.standard_price = cost

    @api.model
    def get_price_from_product(self):
        """ Procesar los productos que tienen el parent_price_product asignado
            Se lanza desde cron despues de que corre el autoload
        """
        prod_obj = self.env['product.template']

        # Busco los que tienen parent price product
        prods = prod_obj.search([('parent_price_product', '!=', False)])
        for prod in prods:
            default_code = prod.parent_price_product
            parent = prod_obj.search([('default_code', '=', default_code)])
            # si ya tiene bien el precio no lo proceso para que no me quede
            # en el historico de precios.
            if parent and parent.list_price != prod.list_price:
                # imaginamos que el costo es la decima parte.
                cost = parent.list_price / 10
                prod.set_prices(cost, 'EFACEC', price=parent.list_price)
                _logger.info('setting price product %s' % prod.default_code)
