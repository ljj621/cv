num_points=[2048, 512, 128]
hidden_channels=64
model=dict(
  type='CompleteDT',
  num_points=num_points,
  hidden_channels=hidden_channels,
  input_conv_layer=dict(
    type='Conv1d',
    channels=[[3, hidden_channels], [hidden_channels, hidden_channels]],
    norm_cfg='BN',
    act_norm='GELU'),
  sampling_layer=dict(type='FPS', num_points=num_points),
  encoder_layer=dict(
    
    
  )
      
 
    
  # )
    
  # dencoder_layer:
  #   generate_points=16384
  #   num_points=*num_points
  #   num_points_layer=3
      

    
)

  
  
    






