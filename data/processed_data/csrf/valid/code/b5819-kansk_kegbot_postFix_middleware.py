# Copyright 2011 Mike Wakerly <opensource@hoho.com>
#
# This file is part of the Pykeg package of the Kegbot project.
# For more information on Pykeg or Kegbot, see http://kegbot.org/
#
# Pykeg is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# Pykeg is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pykeg.  If not, see <http://www.gnu.org/licenses/>.

from pykeg.core import models
from pykeg.web.api.views import check_api_key

from django.core.urlresolvers import reverse
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.template.response import SimpleTemplateResponse
from django.template import RequestContext

ALLOWED_PATHS = (
    '/api/login/',
    '/accounts/login/',
    '/admin/',
    '/media/',
)

def _path_allowed(path, kbsite):
  if kbsite:
    path = path.lstrip(kbsite.url())
  for p in ALLOWED_PATHS:
    if path.startswith(p):
      return True
  return False


class KegbotSiteMiddleware:
  def process_view(self, request, view_func, view_args, view_kwargs):
    """Removes kbsite_name from kwargs if present, and attaches the
    corresponding KegbotSite instance to the request as the "kbsite" attribute.

    If kbsite_name is None, the default site is selected.
    """
    kbsite_name = view_kwargs.pop('kbsite_name', None)
    if not kbsite_name:
      kbsite_name = 'default'
    try:
      request.kbsite = models.KegbotSite.objects.get(name=kbsite_name)
    except models.KegbotSite.DoesNotExist:
      pass
    return None

class SiteActiveMiddleware:
  """Middleware which throws 503s when KegbotSite.is_active is false."""
  def process_view(self, request, view_func, view_args, view_kwargs):
    if not hasattr(request, 'kbsite'):
      return None
    kbsite = request.kbsite

    # We have a KegbotSite, and that site is active: nothing to do.
    if kbsite.is_active:
      return None

    # If the request is for a whitelisted path, allow it.
    <fix/>if _path_allowed(request.path, kbsite):</fix>
      return None

    # Allow staff/superusers access if inactive.
    if request.user.is_staff or request.user.is_superuser:
      return None

    return HttpResponse('Site temporarily unavailable', status=503)

class ApiKeyMiddleware:
  def process_view(self, request, view_func, view_args, view_kwargs):
    if not view_func.__module__.startswith('pykeg.web.api'):
      return None
    if getattr(view_func, 'kb_api_key_required', False):
      check_api_key(request)
      request.kb_api_authenticated = True

class PrivacyMiddleware:
  def process_view(self, request, view_func, view_args, view_kwargs):
    if not hasattr(request, 'kbsite'):
      return None
    elif _path_allowed(request.path, request.kbsite):
      return None
    elif getattr(request, 'kb_api_key_authenticated', False):
      # This is an auth-required kb api view; no need to check privacy since API
      # keys are given staff-level access.
      return None

    privacy = request.kbsite.settings.privacy
    if privacy == 'public':
      return None

    # If non-public, apply the API key check.
    if view_func.__module__.startswith('pykeg.web.api'):
      check_api_key(request)

    if privacy == 'staff' and not request.user.is_staff:
      return SimpleTemplateResponse('kegweb/staff_only.html',
          context=RequestContext(request), status=503)
    elif privacy == 'members':
      if not request.user.is_authenticated or not request.user.is_active:
        return SimpleTemplateResponse('kegweb/members_only.html',
            context=RequestContext(request), status=503)
      return None

    return HttpResponse('Server misconfigured, unknown privacy setting:%s' % privacy, status=503)

