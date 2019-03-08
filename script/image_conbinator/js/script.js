IMG_DIR = 'imgs';

/*
// stylization texture
IMGS = [[
    'camel_ori.png',
    'camel_sty.png'
  ]
];
*/

/* // FDB train
IMGS = [[
  '0000_001original.png',
  '0000_001imgtrain.png',
  '0000_003imgtrain.png',
  '0000_005imgtrain.png',
  '2fdb.png',
  '1fdb.png'
  ],
  [
    '0000_002original.png',
    '0000_002imgtrain.png',
    '0000_004imgtrain.png',
    '0000_006imgtrain.png',
    '3fdb.png',
    '1fdb.png'
  ],
];
*/

/*
IMGS = [[
  'crop_sfn_none_composition_00014.jpg',
  'crop_sfn_featstyle_composition_00014.jpg',
  'crop_sfn_diff_composition_00014.jpg',
  'crop_sfn_none_starrynight_00010.jpg',
  'crop_sfn_featstyle_starrynight_00010.jpg',
  'crop_sfn_diff_starrynight_00010.jpg',
],
  [
    'crop_sfn_none_composition_00015.jpg',
    'crop_sfn_featstyle_composition_00015.jpg',
    'crop_sfn_diff_composition_00015.jpg',
    'crop_sfn_none_starrynight_00011.jpg',
    'crop_sfn_featstyle_starrynight_00011.jpg',
    'crop_sfn_diff_starrynight_00011.jpg',
  ]
];
*/

/*
IMGS = [[
  'sfn_diff_starrynight1_00021.jpg',
  'sfn_diff_candy1_00010.jpg',
  ],
  [
    'sfn_diff_starrynight2_00021.jpg',
    'sfn_diff_candy2_00010.jpg',
  ]
];
*/


IMGS = [[
  '1.png',
  '2.png',
  ],
  [
    '3.png',
    '4.png',
  ]
];


IMG_STYLE = 'height: 120px; margin: 0px';

TABLE_STYLE = {
  border: 0,
  cellspacing: '0px',
}

function setTableStyle(table) {
  for (key in TABLE_STYLE) {
    table.setAttribute(key, TABLE_STYLE[key]);
  }
}

function download(strData, strFileName, strMimeType) {
  console.log("download")
  var D = document,
      A = arguments,
      a = D.createElement("a"),
      d = A[0],
      n = A[1],
      t = A[2] || "text/plain";

  //build download link:
  a.href = "data:" + strMimeType + "," + escape(strData);

  if (window.MSBlobBuilder) {
      var bb = new MSBlobBuilder();
      bb.append(strData);
      return navigator.msSaveBlob(bb, strFileName);
  } /* end if(window.MSBlobBuilder) */

  if ('download' in a) {
      a.setAttribute("download", n);
      a.innerHTML = "downloading...";
      D.body.appendChild(a);
      setTimeout(function() {
          var e = D.createEvent("MouseEvents");
          e.initMouseEvent("click", true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
          a.dispatchEvent(e);
          D.body.removeChild(a);
      }, 66);
      return true;
  } /* end if('download' in a) */
  ; //end if a[download]?

  //do iframe dataURL download:
  var f = D.createElement("iframe");
  D.body.appendChild(f);
  f.src = "data:" + (A[2] ? A[2] : "application/octet-stream") + (window.btoa ? ";base64" : "") + "," + (window.btoa ? window.btoa : escape)(strData);
  setTimeout(function() {
      D.body.removeChild(f);
  }, 333);
  return true;
} /* end download() */

function renderImages(table) {
  IMGS.forEach(row => {
    let tr = document.createElement("tr");
    row.forEach(path => {
      let td = document.createElement("td");
      let img = document.createElement("img");
      img.setAttribute('src', IMG_DIR + '/' + path);
      img.setAttribute('style', IMG_STYLE);
      td.appendChild(img);
      tr.appendChild(td);
    })
    table.appendChild(tr);
  });
}

window.onload = () => {
  console.log("onload");
  html2canvas(document.querySelector("#table"), {
    logging: true,
    useCORS: true,
    onrendered: function (canvas) {       
      console.log("onrendered");     
      img = canvas.toDataURL("image/jpeg");
      download(img, "down.jpg", "image/jpeg");
    }
  }).then(canvas => {
    document.body.appendChild(canvas);
  });
}
