'use strict';

var graph = null;
var mouseOld = {};
var mouseDown = false;
var currentModel = 0;
var loading = false;
var image = null,
    z = null,
    c = null;

var spinner = new Spinner({ color: '#999' });

var COLORS = [
  'black',
  'rgb(208, 2, 27)',
  'rgb(245, 166, 35)',
  'rgb(248, 231, 28)',
  'rgb(139, 87, 42)',
  'rgb(126, 211, 33)',
  'white',
  'rgb(226, 238, 244)',
  'rgb(226, 178, 213)',
  'rgb(189, 16, 224)',
  'rgb(74, 144, 226)',
  'rgb(80, 227, 194)',
];

var MODEL_NAMES = [
  'Anime Face',
  'Human Face',
  //'半身草图',
]

Date.prototype.format = function (format) {
  var o = {
    'M+': this.getMonth() + 1, //month
    'd+': this.getDate(), //day
    'h+': this.getHours(), //hour
    'm+': this.getMinutes(), //minute
    's+': this.getSeconds(), //second
    'q+': Math.floor((this.getMonth() + 3) / 3), //quarter
    S: this.getMilliseconds() //millisecond
  };
  if (/(y+)/.test(format)) format = format.replace(RegExp.$1, (this.getFullYear() + '').substr(4 - RegExp.$1.length));
  for (var k in o) {
    if (new RegExp('(' + k + ')').test(format)) format = format.replace(RegExp.$1, RegExp.$1.length == 1 ? o[k] : ('00' + o[k]).substr(('' + o[k]).length));
  }return format;
};

var MAX_LINE_WIDTH = 10;

function drawShape(x1, y1, x2, y2) {
  if (!image) return;
  graph.drawLine(x1, y1, x2, y2);
}

function getMouse(event) {
  var rect = event.target.getBoundingClientRect();
  var mouse = {
    x: (event.touches && event.touches[0] || event).clientX - rect.left,
    y: (event.touches && event.touches[0] || event).clientY - rect.top
  };
  if (mouse.x > rect.width || mouse.x < 0 || mouse.y > rect.height || mouse.y < 0) return null;
  return mouse;
}

function onMouseDown(event) {
  if (event.button == 2 || loading) {
    mouseDown = false;
    return;
  }
  if (mouseDown) {
    onMouseUp(event);
    return;
  }
  mouseOld = getMouse(event);
  if (mouseOld != null) mouseDown = true;
}

function onMouseUp(event) {
  mouseDown = false;
}

function onMouseMove(event) {
  event.preventDefault();
  if (mouseDown && !loading) {
    var mouse = getMouse(event);
    if (mouse == null) {
      mouseDown = false;
      return;
    }
    drawShape(mouseOld.x, mouseOld.y, mouse.x, mouse.y);
    mouseOld = mouse;
  }
}

function setColor(color) {
  graph.setCurrentColor(color);
  $('#color-drop-menu .color-block').css('background-color', color);
  $('#color-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setLineWidth(width) {
  graph.setLineWidth(width * 2);
  $('#width-label').text(width);
}

function setModel(model) {
  if (loading) return;
  currentModel = model;
  $('#model-label').text(MODEL_NAMES[model]);
  onStart();
}

function setImage(data) {
  setLoading(false);
  if (!data || !data.ok) return;

  if (!image) {
    $('#stroke').removeClass('disabled');
    $('#option-buttons').prop('hidden', false);
  }
  image = data.img;
  z = data.z;
  c = data.c;
  $('#image').attr('src', image);
  $('#canvas').css('background-image', 'url(' + image + ')');
  spinner.spin();
}

function setLoading(isLoading) {
  loading = isLoading;
  $('#srand').prop('disabled', loading);
  $('#submit').prop('disabled', loading);
  $('#spin').prop('hidden', !loading);
  if (loading) {
    $('.image').css('opacity', 0.7);
    spinner.spin(document.getElementById('spin'));
  } else {
    $('.image').css('opacity', 1);
    spinner.stop();
  }
}

function onClear() {
  graph.clear();
}

function onSubmit() {
  if (graph && !loading) {
    setLoading(true);
    var formData = {
      model: MODEL_NAMES[currentModel],
      sketch: graph.getImageData(),
      z: z,
      c: c
    };
    $.post('input', formData, setImage, 'json');
  }
}

function onSrand() {
  if (loading) return;
  setLoading(true);
  onClear();
  $.post('srand', { model: MODEL_NAMES[currentModel] }, setImage, 'json');
}

function onStart() {
  onSrand();
  $('#start').prop('hidden', true);
}

function onUpload() {
  // filename = $('#choose').val();
  // if (!filename) {
  //   alert('未选择文件');
  //   return false;
  // }
  // $.get(filename, function(data) {
  //   console.log(data);
  // });
  // return false;
}

function onChooseFile(e) {
  // filename = $('#choose').val();
  // console.log(filename);
  // $('.custom-file-control').content(filename);
  // console.log($('.custom-file-control').after());
}

function init() {
  COLORS.forEach(function (color) {
    $('#color-menu').append(
      '\n<li role="presentation">\n  <div onclick="setColor(\'' +
      color +
      '\')"\n  >\n    <div class="color-block" style="background-color:' +
      color + ';border:' +
      (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') +
      '"/>\n  </div>\n</li>');
  });

  MODEL_NAMES.forEach(function (model, idx) {
    $('#model-menu').append(
      '<li role="presentation">\n' +
      '  <div class="dropdown-item model-item" onclick="setModel(' + idx + ')">' +
      model +
      '  </div>\n' +
      '</li>\n'
    );
  });

  var slider = document.getElementById('slider');
  noUiSlider.create(slider, {
    start: 5,
    step: 1,
    range: {
      'min': 1,
      'max': 10
    },
    behaviour: 'drag-tap',
    connect: [true, false],
    orientation: 'vertical',
    direction: 'rtl',
  });
  slider.noUiSlider.on('update', function () {
    setLineWidth(parseInt(slider.noUiSlider.get()));
  });

  setColor('black');
  setLineWidth(5);
}

function download(data, filename) {
  var link = document.createElement('a');
  link.href = data;
  link.download = filename;
  link.click();
}

$(document).ready(function () {
  graph = new Graph(document);
  init();

  var canvas = document.getElementById('canvas');
  canvas.addEventListener('mousedown', onMouseDown, false);
  canvas.addEventListener('mouseup', onMouseUp, false);
  canvas.addEventListener('mousemove', onMouseMove, false);
  canvas.addEventListener('mouseout', onMouseUp, false);
  canvas.addEventListener('touchstart', onMouseDown, false);
  canvas.addEventListener('touchend', onMouseUp, false);
  canvas.addEventListener('touchmove', onMouseMove, false);
  canvas.addEventListener('touchcancel', onMouseUp, false);

  $('#download-sketch').click(function () {
    download(canvas.toDataURL('image/png'), 'sketch_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-image').click(function () {
    download(image, 'image_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#clear').click(onClear);
  $('#submit').click(onSubmit);
  $('#stroke').click(function() {
    var stroke = $('#stroke').hasClass('active');
    if (stroke) {
      $('#image').prop('hidden', false);
      $('#canvas').prop('hidden', true);
      $('#stroke').removeClass('active');
      $('#stroke .btn-text').text('显示笔画');
    } else {
      $('#image').prop('hidden', true);
      $('#canvas').prop('hidden', false);
      $('#stroke').addClass('active');
      $('#stroke .btn-text').text('隐藏笔画');
    }
  });
  $('#srand').click(onSrand);
  $('#start').click(onStart);
  // $('#choose').change(onChooseFile);
  // $('#upload').click(onUpload);
});
