IMG_DIR = 'imgs';

IMGS = [[
  'pool_deconv_d_solverx_fake_500.png',
  'pool_deconv_d_solverx_fake_2000.png',
  'pool_deconv_d_solverx_fake_10000.png',
  'pool_deconv_d_solverx_fake_19900.png'
  ],
  [
    'pool_deconv_d_solverx_fake_400.png',
    'pool_deconv_d_solverx_fake_1900.png',
    'pool_deconv_d_solverx_fake_9900.png',
    'pool_deconv_d_solverx_fake_19800.png'
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
