	
	
	
$(function(){

	/*變量宣告*/
	var i = 1;
	var playTime = 1000;//預設動畫播放時間1000ms完成動畫效果
	

	/*當點擊 #mybox, 則呼叫function*/
	$("#Box-mainContance_Home-Chat_My").click(function(){

		var Function_Link_A = document.getElementById("Box-mainContance_Home-Chat_A");
		var Function_Link_B = document.getElementById("Box-mainContance_Home-Chat_B");
		var Function_Link_C = document.getElementById("Box-mainContance_Home-Chat_C");
		var Function_Link_D = document.getElementById("Box-mainContance_Home-Chat_D");
		var Function_My_Data = document.getElementById("Box-mainContance-Home-Chat_My-Data");

		if(i%2 == 0){
			$(this).animate({//原本位置的座標//回覆原來樣子
				"left":'650px',
				"top":'250px'
			},playTime);

			$(Function_Link_A).animate({
				"opacity":'1',
				"z-index":'6'
			},playTime);

			$(Function_Link_B).animate({
				"opacity":'1',
				"z-index":'6'
			},playTime);

			$(Function_Link_C).animate({
				"opacity":'1',
				"z-index":'6'
			},playTime);

			$(Function_Link_D).animate({
				"opacity":'1',
				"z-index":'6'
			},playTime);

			$(Function_My_Data).animate({
				"opacity":'0',
				"z-index":'-1'
			},playTime-500);

		}else{
			$(this).animate({//目的地座標
				"left":'300px',
				"top":'30px'
			},playTime);

			$(Function_Link_A).animate({
				"opacity":'0',
				"z-index":'0'
			},playTime);

			$(Function_Link_B).animate({
				"opacity":'0',
				"z-index":'0'
			},playTime);

			$(Function_Link_C).animate({
				"opacity":'0',
				"z-index":'0'
			},playTime);

			$(Function_Link_D).animate({
				"opacity":'0',
				"z-index":'0'
			},playTime);

			$(Function_My_Data).animate({
				"z-index":'0',
				"opacity":'1',
			},playTime);
		}
		i++;

		/*防止var‘i’溢位*/
		if(i >= 10){
			i = 0;
		}//點擊等於10次則reset i的值（避免多次按下導致var溢位）
	});
})
