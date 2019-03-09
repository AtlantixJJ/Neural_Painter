# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.template import loader, Context
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import home.api as api

from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode

INDEX_FILE = "index_en.html"

def response(image, z, c):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    imageString = b64encode(buffered.getvalue()).decode('utf-8')
    z = b64encode(z).decode('utf-8')
    c = b64encode(c).decode('utf-8')
    json = '{"ok":"true","img":"data:image/png;base64,%s","z":"%s","c":"%s"}' % (
        imageString, z, c)
    return HttpResponse(json)


def index(request):
    return render(request, INDEX_FILE)

@csrf_exempt
def input_image(request):
    form_data = request.POST
    #print(form_data)
    if request.method == 'POST' and 'sketch' in form_data and 'model' in form_data:
        try:
            model = form_data['model']
            if not api.model_exist(model):
                return HttpResponse('{}')

            imageData = b64decode(form_data['sketch'].split(',')[1])
            z = b64decode(form_data['z'])
            c = b64decode(form_data['c'])

            image = Image.open(BytesIO(imageData))
            [sketch, mask] = api.get_array(model, image)
            #print("=> edit")
            gen, z, c = api.generate_image(model, sketch, mask, z, c)
            #print("=> edit done")

            return response(gen, z, c)
        except Exception as e:
            print("!> Exception:")
            print(e)
            return HttpResponse('{}')
    return HttpResponse('{}')


@csrf_exempt
def srand(request):
    form_data = request.POST
    #print(form_data)
    if request.method == 'POST' and 'model' in form_data:
        try:
            model = form_data['model']
            if not api.model_exist(model):
                return HttpResponse('{}')

            gen, z, c = api.regenerate_image(model)
            return response(gen, z, c)
        except Exception as e:
            print("!> Exception:")
            print(e)
            return HttpResponse('{}')
    return HttpResponse('{}')
